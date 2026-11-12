# RobustNASMetric Repository Reconciliation Design

## Intent

Make `https://github.com/SupCE-moon/RobustNASMetric` the single collaboration
and release center for this research project, without losing work currently
split between local Git worktrees, the legacy GitHub repository, and the
`wangjiahao` server container.

## Goals

- Preserve all meaningful Git history already available locally.
- Import server-side source code, documentation, and small reproducibility
  artifacts that are not already represented locally.
- Preserve the server as the storage location for datasets, model weights,
  checkpoints, adversarial caches, and large raw outputs.
- Record enough provenance for every retained server result to be located and
  verified later.
- Establish durable handoff documentation so Codex, DSH, and the user can
  continue independently.

## Non-goals

- Do not delete, overwrite, or reorganize the existing server deployment.
- Do not commit model weights, datasets, checkpoints, virtual environments,
  caches, secrets, or large raw experiment files.
- Do not force-push, rewrite historical commits, or replace the legacy
  repository.
- Do not merge this reconciliation directly into `main` before review.

## Source Roles

| Source | Role |
|---|---|
| New GitHub repository `SupCE-moon/RobustNASMetric` | Future canonical collaboration and release repository. |
| Legacy GitHub repository `cduai/robust_metric` | Read-only historical reference after migration. |
| Local Git worktrees | Source for existing commits and uncommitted work; only committed, reviewed material is eligible for migration. |
| Server `/workspace/robust_metric_lab_nas` | Deployment snapshot and authoritative location of heavyweight outputs. Its recorded base is commit `dc6fb9cfb2f6bd95dcb43acaf01d7161d92fc3e5`; it is not itself a Git worktree. |

## Repository Content Policy

Track in Git:

- source, scripts, configurations, tests, and documentation;
- small result tables and publication-ready figures;
- manifests that name server artifacts, execution commands, timestamps,
  sizes, and SHA-256 checksums where practical;
- project handoff, decision, and status documents.

Keep only on the server:

- datasets, weights, checkpoints, virtual environments, and adversarial
  caches;
- raw artifacts too large for source control;
- ephemeral logs and temporary staging outputs.

## Migration Safety Rules

1. Reconcile on a dedicated branch before changing `main` or a remote default
   branch.
2. Compare file contents; do not overwrite a local file merely because a
   server file has a newer timestamp.
3. Copy only the server files selected by an explicit inventory.
4. Inspect every candidate for credentials and size before adding it to Git.
5. Use a remote named `new-origin` for the new repository during review;
   retain `origin` as the legacy remote until the migration is accepted.
6. Push a review branch first. Only change the default working remote after
   the user reviews the reconciliation and its GitHub contents.
7. Snapshot the Git state and uncommitted file inventory of every local
   worktree before selecting material for migration. A source or documentation
   file must be explicitly included or explicitly excluded; it cannot vanish
   silently.
8. Warn when a proposed tracked file is 10 MiB or larger and reject files
   larger than 50 MiB unless the user explicitly approves an exception.
9. Scan staged text content for credentials before every migration commit and
   verify the `new-origin` URL immediately before the first push.

## Handoff Contract

The migrated repository will contain:

- `docs/HANDOFF.md`: active task, environment and server access conventions,
  reproducibility commands, and rules for handoff;
- `docs/PROJECT_STATUS.md`: current experiment state, next priority, and
  verified conclusions;
- `docs/DECISIONS.md`: frozen protocol decisions and exclusions;
- `results/manifests/`: provenance records for server-resident artifacts.

Every future task must work on a named branch, record verification performed,
and update the handoff/status documents before it is handed off.
