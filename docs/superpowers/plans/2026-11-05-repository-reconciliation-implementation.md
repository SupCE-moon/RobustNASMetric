# Repository Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely establish `SupCE-moon/RobustNASMetric` as the project’s collaboration repository while retaining provenance for local and server-only research work.

**Architecture:** A dedicated reconciliation branch records an immutable inventory before importing any selected material. Local Git history is retained, server-only source/doc files are compared by content before import, and heavyweight artifacts remain on the server with checked manifests. The new GitHub remote receives only a reviewed branch; the legacy remote remains unchanged until approval.

**Tech Stack:** Git, PowerShell, SSH/OpenSSH, Python/pytest, SHA-256.

**Spec:** `docs/superpowers/specs/2026-11-05-repository-reconciliation-design.md`

## Global Constraints

- Do not alter the primary dirty worktree or any server file.
- Retain legacy `origin`; add `new-origin` only for `https://github.com/SupCE-moon/RobustNASMetric.git`.
- Do not force-push, rewrite history, or push directly to a default branch.
- Track only source, tests, documentation, small results, and manifests.
- Warn at 10 MiB and reject artifacts larger than 50 MiB without explicit approval.
- Never commit credentials, datasets, weights, checkpoints, virtual environments, caches, or raw large artifacts.

## Review Focus

- A modified or untracked file in any local worktree must be classified as imported, excluded, or snapshot-only before migration.
- A newer server timestamp must not overwrite a different local file without a SHA-256/content comparison.
- Binary weights, archives, caches, and raw outputs must be rejected even when their filename is misleading.
- A staged secret-like value must block a commit rather than merely being reported.
- The first push must target the supplied new repository and a reconciliation branch, never the legacy remote or a default branch.

---

### Task 1: Create a read-only source inventory

**Files:**
- Create: `results/manifests/reconciliation-2026-11-05/local-worktrees.json`
- Create: `results/manifests/reconciliation-2026-11-05/server-inventory.json`
- Create: `results/manifests/reconciliation-2026-11-05/README.md`

**Interfaces:**
- Consumes: local Git worktree metadata and read-only server file metadata.
- Produces: JSON inventories with path, source, type, byte count, mtime, SHA-256 where feasible, and classification.

- [ ] **Step 1: Inventory every local worktree and its dirty state**

Run: `git worktree list --porcelain` and record each worktree’s HEAD, branch, tracked diff names, and untracked file names without staging or modifying them.

- [ ] **Step 2: Inventory candidate server files read-only**

Record selected server source, documentation, result-summary, and heavyweight artifact paths under `/workspace/robust_metric_lab_nas`, including metadata and SHA-256 for candidate imported files.

- [ ] **Step 3: Classify every candidate**

Use only `import`, `excluded`, or `server-only` classifications. Give each excluded/server-only file a reason.

- [ ] **Step 4: Verify inventory completeness**

Run: `python -m json.tool results/manifests/reconciliation-2026-11-05/local-worktrees.json` and repeat for `server-inventory.json`.

- [ ] **Step 5: Commit the inventory**

```bash
git add results/manifests/reconciliation-2026-11-05
git commit -m "docs: inventory reconciliation sources"
```

### Task 2: Establish repository policy and handoff records

**Files:**
- Create: `docs/HANDOFF.md`
- Create: `docs/PROJECT_STATUS.md`
- Create: `docs/DECISIONS.md`
- Modify: `.gitignore`
- Test: `tests/test_repository_policy.py`

**Interfaces:**
- Consumes: Task 1 inventories and the reconciliation design.
- Produces: documented collaboration contract and machine-checked ignore/size policy.

- [ ] **Step 1: Write failing policy tests**

```python
def test_repository_policy_rejects_weight_and_large_artifact(tmp_path):
    ...

def test_repository_policy_accepts_small_csv_manifest(tmp_path):
    ...
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_repository_policy.py -v`
Expected: FAIL because the policy helper does not exist.

- [ ] **Step 3: Implement a small policy helper in `tools/repository_policy.py`**

Expose `classify_path(path: Path, size_bytes: int) -> str`, returning `reject`, `warn`, or `allow`; reject prohibited extensions/categories and files larger than 50 MiB, warn at 10 MiB.

- [ ] **Step 4: Write handoff documents and tighten `.gitignore`**

Document the server path, branch rule, provenance requirement, result policy, and the rule that secrets never enter Git. Ignore large ephemeral output trees while preserving `results/manifests/` and selected small summaries.

- [ ] **Step 5: Verify policy and documentation**

Run: `pytest tests/test_repository_policy.py -v`
Expected: PASS.

- [ ] **Step 6: Commit the policy baseline**

```bash
git add .gitignore docs/HANDOFF.md docs/PROJECT_STATUS.md docs/DECISIONS.md tools/repository_policy.py tests/test_repository_policy.py
git commit -m "docs: establish repository handoff policy"
```

### Task 3: Import reviewed server-only source and documentation

**Files:**
- Modify: selected paths classified `import` in Task 1 only
- Create: `results/manifests/reconciliation-2026-11-05/imported-files.json`
- Test: `tests/test_repository_policy.py`

**Interfaces:**
- Consumes: Task 1 inventory and Task 2 policy helper.
- Produces: a content-verified, reviewable import with an import manifest.

- [ ] **Step 1: Compare selected files by SHA-256 and text diff**

For each candidate, preserve both versions when content differs and only one version cannot be identified as superseded from documented provenance.

- [ ] **Step 2: Copy only classified `import` source/doc/small-summary files**

Do not import weights, archives, virtual environments, raw large output trees, or any file marked `server-only`/`excluded`.

- [ ] **Step 3: Record imported file provenance**

Write source server path, source SHA-256, destination path, and inclusion reason to `imported-files.json`.

- [ ] **Step 4: Run safety verification**

Run: `pytest tests/test_repository_policy.py -v` and inspect `git diff --check`.
Expected: PASS and no whitespace errors.

- [ ] **Step 5: Commit the reviewed import**

```bash
git add <reviewed-import-paths> results/manifests/reconciliation-2026-11-05/imported-files.json
git commit -m "chore: import reconciled server sources"
```

### Task 4: Publish a review branch to the new repository

**Files:**
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: reviewed commits from Tasks 1–3.
- Produces: a remote reconciliation branch and documented verification result.

- [ ] **Step 1: Add and inspect the review remote**

Run: `git remote add new-origin https://github.com/SupCE-moon/RobustNASMetric.git` only if it is absent, then `git ls-remote new-origin`.

- [ ] **Step 2: Scan staged and committed text for secrets**

Run a documented scan over files introduced on the reconciliation branch. Any credential-like match blocks the push until removed.

- [ ] **Step 3: Verify target and branch**

Run: `git remote get-url new-origin` and `git branch --show-current`.
Expected: the supplied GitHub URL and `codex/repository-reconciliation`.

- [ ] **Step 4: Push only the reconciliation branch**

Run: `git push -u new-origin codex/repository-reconciliation`.
Expected: a new remote review branch; no changes to the legacy `origin` or remote default branch.

- [ ] **Step 5: Record the published commit and review URL**

Update handoff/status documents with the commit ID, remote branch, verification commands, and outstanding server-only artifacts; commit the record.

### Task 5: Review and adopt the new repository

**Files:**
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: the remote reconciliation branch from Task 4.
- Produces: a user-approved migration decision.

- [ ] **Step 1: Review the GitHub branch contents with the user**

Confirm that required code/docs are present, heavyweight files are absent, and manifests point to the intended server artifacts.

- [ ] **Step 2: Merge through GitHub after explicit approval**

Merge the reviewed branch using the user’s preferred GitHub workflow; do not force-push or alter the legacy repository.

- [ ] **Step 3: Change the default collaboration convention**

After merge, document `SupCE-moon/RobustNASMetric` as the canonical collaboration remote and `cduai/robust_metric` as legacy read-only history.
