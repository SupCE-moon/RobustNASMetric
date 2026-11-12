"""Read-only inventory of the remote project tree, over SSH.

The server is not a Git worktree, so its content has to be captured by hand
before anything can be imported: every file needs an absolute path, size,
modification time and (where affordable) a SHA-256.  This tool produces that
census and nothing else -- it never writes, moves or deletes anything remotely.

The password is taken from ``ROBUST_METRIC_SSH_PASSWORD`` or prompted for; it is
never accepted as a command-line argument and never written to disk.

Example::

    $env:ROBUST_METRIC_SSH_PASSWORD = "<password>"
    python tools/server_inventory.py --host 49.232.181.138 --port 17010 \
        --user guzhengkai --docker wangjiahao \
        --output results/manifests/reconciliation-20261112/server-wangjiahao.json
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repository_policy import classify_path  # noqa: E402

SCHEMA_VERSION = 1

#: Executed on the server with ``python3 -``; reads the tree, prints one JSON blob.
REMOTE_PROGRAM = r'''
import hashlib, json, os, sys
from datetime import datetime, timezone

root = sys.argv[1]
hash_limit = int(sys.argv[2])
skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules",
             ".pytest_cache", ".mypy_cache", ".ruff_cache"}

def sha256(path, size):
    if hash_limit >= 0 and size > hash_limit:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

entries = []
errors = []
for current, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
    for name in sorted(filenames):
        absolute = os.path.join(current, name)
        relative = os.path.relpath(absolute, root).replace(os.sep, "/")
        try:
            stat = os.stat(absolute)
            if not os.path.isfile(absolute):
                continue
            entries.append({
                "path": relative,
                "absolute_path": absolute,
                "size": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256(absolute, stat.st_size),
            })
        except OSError as error:
            errors.append({"path": relative, "error": str(error)})

top = {}
for entry in entries:
    key = entry["path"].split("/", 1)[0]
    bucket = top.setdefault(key, {"files": 0, "bytes": 0})
    bucket["files"] += 1
    bucket["bytes"] += entry["size"]

print(json.dumps({
    "root": root,
    "hash_limit_bytes": hash_limit,
    "host_python": sys.version.split()[0],
    "summary": {
        "files": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "hashed": sum(1 for e in entries if e["sha256"]),
    },
    "top_level": top,
    "errors": errors,
    "entries": entries,
}, ensure_ascii=False))
'''


def connect(host: str, port: int, user: str, password: str, timeout: int = 20):
    import paramiko

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run_remote(client, command: str, timeout: int) -> tuple[str, str, int]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return out, err, stdout.channel.recv_exit_status()


def build_command(root: str, hash_limit: int, container: str | None, python: str) -> str:
    program = REMOTE_PROGRAM
    if container:
        # Feed the program through a here-document so quoting never breaks.
        return (
            f"docker exec -i {container} python3 - {root} {hash_limit} <<'PY'\n"
            f"{program}\nPY"
        )
    return f"{python} - {root} {hash_limit} <<'PY'\n{program}\nPY"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--root", default="/workspace/robust_metric_lab_nas")
    parser.add_argument("--docker", default=None, help="run the census inside this container")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-limit-mb", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    password = os.environ.get("ROBUST_METRIC_SSH_PASSWORD") or getpass.getpass(
        f"SSH password for {args.user}@{args.host}: "
    )
    if not password:
        print("no password supplied", file=sys.stderr)
        return 2

    client = connect(args.host, args.port, args.user, password)
    try:
        command = build_command(
            args.root, args.hash_limit_mb * 1024 * 1024, args.docker, args.python
        )
        out, err, status = run_remote(client, command, args.timeout)
    finally:
        client.close()

    if status != 0 or not out.strip().startswith("{"):
        print(f"remote census failed (exit {status}): {err[:2000]}", file=sys.stderr)
        return 1

    remote = json.loads(out)
    for entry in remote["entries"]:
        entry["classification"] = classify_path(entry["path"], entry["size"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "transport": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "container": args.docker,
        },
        "excluded_dirs": sorted(
            {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
        ),
        "rejected": sum(
            1 for entry in remote["entries"] if entry["classification"] == "reject"
        ),
        **remote,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"server census: {remote['summary']['files']} files, "
        f"{remote['summary']['total_bytes'] / 1073741824:.2f} GiB, "
        f"{remote['summary']['hashed']} hashed, {payload['rejected']} rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
