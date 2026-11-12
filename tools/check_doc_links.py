"""Check that paths mentioned in the docs actually exist.

Why this exists
---------------
"详见 `results/audits/xxx.md`" is the project's promise of traceability. The
promise is only worth something if the file is there. Over months, reports get
renamed, audits move, and a reader who follows a dead path stops trusting the
report. This tool walks the Markdown, pulls out every repository-relative path
that is written in backticks or as a markdown link, and checks existence.

Why ``outputs/`` is not an error
--------------------------------
Most of ``outputs/`` is gitignored on purpose (weights, caches, 14 MiB raw
tables). A doc pointing into ``outputs/`` is describing where a result was
produced on the machine that produced it, not claiming it ships with the repo.
Those paths are reported separately as "not in the repo" so that a genuine typo
inside ``outputs/`` still shows up, without drowning the real breakage.

Usage
-----
    python tools/check_doc_links.py \
        --json results/audits/doc-links-20261112.json \
        --report results/audits/doc-links-20261112.md
    # --strict: exit 1 when a repository path is missing (for a future CI gate)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Directories whose contents are supposed to ship with the repository.
REQUIRED_ROOTS = ("docs", "src", "scripts", "tests", "tools", "configs", "results")
# Produced artifacts; mostly gitignored, so absence is expected, not broken.
OPTIONAL_ROOTS = ("outputs",)

SUFFIXES = (
    ".md",
    ".py",
    ".csv",
    ".json",
    ".patch",
    ".txt",
    ".log",
    ".sh",
    ".mjs",
    ".yaml",
    ".yml",
    ".toml",
)

BACKTICK = re.compile(r"`([^`\n]+)`")
MD_LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)\)")

# Design documents describe files that were *going to* be written, and executors
# renamed things along the way (e.g. run_activation_holdout.py became
# train_activation_holdout_nb201.py). A stale name there is history, not a bug,
# so it is reported in its own bucket instead of as breakage.
PLAN_PREFIXES = ("docs/superpowers/plans/", "docs/superpowers/specs/")
# Files that belong to an external delivery bundle, mentioned here for context.
EXTERNAL_PREFIXES = ("docs/student_robust_metric_benchmark/",)

# (doc, path) -> why the path is legitimately absent from this repository.
ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "docs/experiments/activation_holdout_provenance_audit_20261027.md",
        "scripts/train_activation_holdout_nb201_v2.py",
    ): "文档写明这是「服务器新文件」，与本地 train_activation_holdout_nb201.py 字节相同",
    (
        "docs/experiments/student_benchmark_issue_log.md",
        "docs/experiments/source_snapshot/verifier_after_recovery.log",
    ): "文档写明日志保存在容器项目内，不是本仓库文件",
    (
        "docs/experiments/教师交接README服务器核对_20261031.md",
        "scripts/verify_student_benchmark_bundle.py",
    ): "教师交付包内自带的验收脚本，随交付包提供，不属于本项目仓库",
    (
        "docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md",
        "scripts/audit_phi_incremental.py",
    ): "方案文档里计划创建的脚本，批准并创建后这条自动失效",
    (
        "docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md",
        "tests/test_audit_phi_incremental.py",
    ): "方案文档里计划创建的测试，批准并创建后这条自动失效",
}



def doc_files(root: Path) -> list[Path]:
    files = [root / "README.md"]
    files += sorted((root / "docs").rglob("*.md"))
    # This tool writes its own report into results/audits/, and that report lists
    # paths. Scanning it would feed our own output back in as "broken links"
    # (every path it just told you about would look missing on the next run).
    files += [
        path
        for path in sorted((root / "results").rglob("*.md"))
        if not path.name.startswith("doc-links-")
    ]
    return [path for path in files if path.is_file()]


def normalize(candidate: str) -> str | None:
    """Return a repo-relative POSIX path, or None if this is not one of ours."""
    text = candidate.strip().strip("`'\"")
    text = text.split("#", 1)[0].strip()
    if not text or " " in text or "<" in text or ">" in text:
        return None
    if "*" in text or text.startswith(("http://", "https://", "/", "~", "C:")):
        return None
    text = text.replace("\\", "/")
    if not text.endswith(SUFFIXES):
        return None
    root = text.split("/", 1)[0]
    if root not in REQUIRED_ROOTS + OPTIONAL_ROOTS:
        return None
    return text


def scan(root: Path) -> dict:
    rows: dict[str, list[dict]] = {
        "missing_required": [],
        "missing_planned": [],
        "missing_external": [],
        "missing_explained": [],
        "missing_optional": [],
    }
    checked = 0

    for path in doc_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        doc = path.relative_to(root).as_posix()
        seen: set[str] = set()
        candidates = [m.group(1) for m in BACKTICK.finditer(text)]
        candidates += [m.group(1) for m in MD_LINK.finditer(text)]
        for raw in candidates:
            target = normalize(raw)
            if target is None or target in seen:
                continue
            seen.add(target)
            checked += 1
            if (root / target).exists():
                continue
            record = {"doc": doc, "path": target}
            if (doc, target) in ALLOWLIST:
                record["reason"] = ALLOWLIST[(doc, target)]
                rows["missing_explained"].append(record)
            elif target.startswith(EXTERNAL_PREFIXES):
                rows["missing_external"].append(record)
            elif doc.startswith(PLAN_PREFIXES):
                rows["missing_planned"].append(record)
            elif target.split("/", 1)[0] in OPTIONAL_ROOTS:
                rows["missing_optional"].append(record)
            else:
                rows["missing_required"].append(record)

    for bucket in rows.values():
        bucket.sort(key=lambda item: (item["doc"], item["path"]))
    return {
        "schema_version": 1,
        "docs_scanned": len(doc_files(root)),
        "paths_checked": checked,
        **rows,
    }


def _table(items: list[dict], *, with_reason: bool = False) -> list[str]:
    if not items:
        return ["无。"]
    head = "| 文档 | 引用的路径 | 原因 |" if with_reason else "| 文档 | 引用的路径 |"
    lines = [head, "| --- | --- | --- |" if with_reason else "| --- | --- |"]
    for item in items:
        if with_reason:
            lines.append(f"| `{item['doc']}` | `{item['path']}` | {item['reason']} |")
        else:
            lines.append(f"| `{item['doc']}` | `{item['path']}` |")
    return lines


def render_markdown(payload: dict) -> str:
    required = payload["missing_required"]
    planned = payload["missing_planned"]
    external = payload["missing_external"]
    explained = payload["missing_explained"]
    optional = payload["missing_optional"]
    lines = [
        "# 文档里的路径引用：存在性检查",
        "",
        f"- 扫描文档：{payload['docs_scanned']} 份",
        f"- 检查过的仓库内路径：{payload['paths_checked']} 条",
        f"- 应当修复的缺失（`docs/ src/ scripts/ tests/ tools/ configs/ results/`）：**{len(required)}** 条",
        "",
        "## 1. 应当修复的缺失",
        "",
        "这些文档承诺「详见某文件」，但文件不在仓库里。这是唯一一组算错误的。",
        "",
        *_table(required),
        "",
        "## 2. 计划/设计文档里的历史文件名",
        "",
        "`docs/superpowers/plans/` 与 `specs/` 记录的是**当时打算创建**的文件，实际实现常改名"
        "（例如 `run_activation_holdout.py` 最终叫 `train_activation_holdout_nb201.py`）。"
        "这些是历史，不是断链，但列出来便于对照。",
        "",
        *_table(planned),
        "",
        "## 3. 外部交付包内的路径",
        "",
        *_table(external),
        "",
        "## 4. 已在文档里说明为「服务器/容器上的文件」的引用",
        "",
        "文档自己写明该文件不在本仓库，因此不算断链；记录原因以免日后被误删理由。",
        "",
        *_table(explained, with_reason=True),
        "",
        "## 5. `outputs/` 下的缺失（信息性）",
        "",
        "这些是「在生成结果的机器上存在」的路径，按策略不入库。列在这里是为了让",
        "真正的拼写错误仍能被看见，而不是为了让它们全部变绿。",
        "",
        *_table(optional),
        "",
        "## 这条检查能证明什么",
        "",
        "- 能证明：文档里指向仓库内文件的每一处引用，指到的文件**确实在**。读者顺着"
        "「详见 xxx」走下去不会撞空。",
        "- 不能证明：文件内容与文档描述一致——那要靠 `verify_doc_claims.py`（数字）和人工阅读（论述）。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = scan(args.root.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    args.report.write_text(render_markdown(payload), encoding="utf-8")

    print(
        f"docs={payload['docs_scanned']} checked={payload['paths_checked']} "
        f"missing_required={len(payload['missing_required'])} "
        f"planned={len(payload['missing_planned'])} "
        f"external={len(payload['missing_external'])} "
        f"explained={len(payload['missing_explained'])} "
        f"optional={len(payload['missing_optional'])}"
    )
    if args.strict and payload["missing_required"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
