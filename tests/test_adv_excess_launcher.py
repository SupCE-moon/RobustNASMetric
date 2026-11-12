"""Regression coverage for the GPU gate shell function."""

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASH = "D:/DevelopmentTools/git/Git/bin/bash.exe" if os.name == "nt" else "bash"


def test_gpu_gate_returns_success_after_three_free_checks(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    source = (ROOT / "scripts/run_adv_excess_full.sh").read_text(encoding="utf-8")
    function = source.split("wait_for_gpu() {", 1)[1].split("\n}\n", 1)[0]
    script = (
        "set -euo pipefail\n"
        "device=cuda\n"
        f"run_dir='{tmp_path.as_posix()}'\n"
        "nvidia-smi() { [[ \"$*\" == *--query-gpu=memory.used* ]] || return 9; echo 2; }\n"
        "sleep() { :; }\n"
        "wait_for_gpu() {" + function + "\n}\n"
        "wait_for_gpu\n"
        "echo passed\n"
    )
    result = subprocess.run(
        [BASH, "-c", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout


def test_gpu_gate_does_not_count_a_failed_monitor_query_as_free(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    source = (ROOT / "scripts/run_adv_excess_full.sh").read_text(encoding="utf-8")
    function = source.split("wait_for_gpu() {", 1)[1].split("\n}\n", 1)[0]
    script = (
        "set -euo pipefail\n"
        "device=cuda\n"
        f"run_dir='{tmp_path.as_posix()}'\n"
        "nvidia-smi() {\n"
        "  if [[ ! -e \"$run_dir/first_query_failed\" ]]; then\n"
        "    touch \"$run_dir/first_query_failed\"; return 1\n"
        "  fi\n"
        "  echo 2\n"
        "}\n"
        "sleep() { :; }\n"
        "wait_for_gpu() {" + function + "\n}\n"
        "wait_for_gpu\n"
    )
    result = subprocess.run(
        [BASH, "-c", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    checks = (tmp_path / "STATUS.env").read_text(encoding="utf-8").splitlines()
    assert len(checks) == 4
    assert "query_failed" in checks[0]
