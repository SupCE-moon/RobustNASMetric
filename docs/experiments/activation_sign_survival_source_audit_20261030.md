# Activation sign survival source audit (2026-10-30)

Status: canonical source mirror verified; local baseline passes. No candidate calculation or new ranking.

- Allowed scope: `wangjiahao` container, `/workspace/robust_metric_lab_nas`.
- `strict_adv_excess_release_20261024T110422Z/release_verification.json`: PASS, 48,160 unique raw keys, 84 ranked metrics, 2 excluded.
- Server project is not a Git repository (`git status` fails at filesystem boundary); freeze individual source hashes instead of inferring a server commit.
- `student_benchmark.py`: `7dd95096c54b5d095ad98065897361ac081822762272d1cb796e7112a555306b`.
- `student_runner.py`: `014f9e3276be11228ce461d1a685711e65f71ba18053fa8ef6bd80ebdd8ab835`.
- `evaluate_rb40_student.py`: `f41edfd48e734e28bb45838558a620abb1b7c32336141b660b125835d919eee5`.
- `evaluate_nb201_student.py`: `fa20c2461a440cb52365c160b6c23896ab78bfddaadcabbd5f7089c28120539e`.
- `evaluate_darts100_student.py`: `f6589bede6e4ffe632671770acce130169c67911bb4a87c4e7d7fe099d8389f8`.

These five files were absent from the local isolated worktree at the start of this audit. Exact copies were transferred from the mounted wangjiahao project, and all five local SHA-256 digests matched the container values above. The transfer changed no server file. With `PYTHONPATH=src`, the full local suite now passes **105/105** when run outside the restrictive temporary-file sandbox. Inside that sandbox, 104 passed and one unrelated NB201 checkpoint test failed on a Windows temporary-file rename permission error; this was not a model or metric failure.

The authoritative loader entrypoints are `load_rb40_items`, `load_nb201_items`, `load_darts_items` in `student_benchmark.py`; `evaluate_items` and `load_cifar10_samples` in `student_runner.py`; and `make_model_loader` in each of the three copied evaluator scripts. The server project has no Git metadata, so the five file hashes—not an inferred server commit—are the source attestation. Before GPU work, the remaining input manifests, sample file, checkpoints, strict joiner and resource-monitor source still need their hashes frozen.

The prior full-run manifest pins the three shared benchmark CSVs at `shared_benchmarks/robust_metric_v1/manifests/`: RB40 `5b51979988249284e14106c0f1121b421a7ce147c4dff3c32b30f2120bc13a08b`, NB201 `a158c609a84f44789153870019eb593c264eb71cb5d83eeac496185f3adf8976`, DARTS `7761b4a4899a76734d8d447fa5aacd08ed08c33cf8db2e0206a483d290314b12`. These are historical source hashes; the new run must rehash current files before freezing.

Strict tooling hashes observed in-container: `build_adv_excess_strict_release.py` `a346c9110a6ebbcc0dfeea09c0fdb101b32eaa123cfba9e1e5af2e35154bc19a`; `verify_strict_release.py` `4d6e603a3eccc2bb61360b43e38fe4083088663a3d8829585e81f300df35de6e`; `build_strict_metric_ranking.py` `1b01e216c9c773c7e6654eb458dd0a57244a31d73a58df1705affb08f7e2f975`. Existing resource-wait script `wait_and_run_strict_smoke.sh` hashes to `2867c08c32aaa1cbe3c8e41ed02762fab91c9e71be22c7f61fdd88a7e2d437f5`; inspect its threshold and scope before reuse. No script was executed by this audit.

2026-10-30 16:11 UTC, 为新候选的严格发布入口补镜像四份服务器源码；SFTP 只读提取后，本地 SHA-256 与 `wangjiahao` 内逐份相同：`strict_ranking.py` `ab1cff50f13fcfc536258ca73717614ab2cf7e21c1715243dc0fabd4b83a04c1`；`release.py` `e71b51684bee6837b5b7f790efe797f07a8f387772b2c9bbf07f45cbe660d2eb`；`metric_eligibility.py` `e03131838aec842ae8edde806b07df50b59cac4b8d788c80975b23a6f8a972a8`；`student_analysis.py` `e765d50d10c77ca77a4d462d81960c25170029e2dad0ede34239080861152b1f`。镜像后本地完整测试 148/148 通过。此操作未修改服务器源码；当前冻结任务仍在读取权重。

The rank source currently places TRNAS Phi first (`all_setting_mean=0.7822227242461846`); E37 multiview max and RMS are sixth/seventh. Historical releases remain read-only.
