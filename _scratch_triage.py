"""Which branch's metrics.py is the superset? Function names decide it, not byte counts."""

import ast
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

REVS = {
    "main(784fc74)": "784fc74",
    "server-mainline(dc6fb9c)": "dc6fb9c",
    "strict-e37(893a994)": "893a994",
    "activation(3741aab?)": "dfc67f2",
    "three-space(2bcb95c)": "2bcb95c",
}

def functions(rev: str, path: str) -> set[str]:
    text = subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="replace")
    tree = ast.parse(text)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names

for path in ("src/robust_metric_lab/metrics.py",):
    sets = {label: functions(rev, path) for label, rev in REVS.items()}
    for label, names in sets.items():
        print(f"{label:28s} {len(names):3d} functions")
    print()
    base = sets["main(784fc74)"]
    server = sets["server-mainline(dc6fb9c)"]
    strict = sets["strict-e37(893a994)"]
    print("server vs main  : missing_from_server =", sorted(base - server))
    print("strict vs server: missing_from_strict =", sorted(server - strict))
    print("server vs strict: missing_from_server =", sorted(strict - server))
