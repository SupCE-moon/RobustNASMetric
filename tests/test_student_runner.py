from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robust_metric_lab.student_benchmark import BenchmarkManifestItem
from robust_metric_lab.student_benchmark import load_nb201_items
from robust_metric_lab.student_runner import (
    CSV_COLUMNS,
    append_rows_atomic,
    evaluate_items,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "scripts" / "evaluate_rb40_student.py",
    ROOT / "scripts" / "evaluate_nb201_student.py",
    ROOT / "scripts" / "evaluate_darts100_student.py",
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-1",
        "benchmark": "nb201",
        "setting": "nb201_e5",
        "model_or_arch_id": "a",
        "checkpoint_epoch": 5,
        "metric_name": "feature_delta",
        "direction": "lower",
        "metric_value": 1.25,
        "target_value": 0.7,
        "status": "ok",
        "error": "",
        "runtime_sec": 0.01,
        "seed": 20260808,
        "sample_offset": 0,
        "num_samples": 64,
        "epsilon": 8 / 255,
        "checkpoint_path": str(Path.cwd().resolve() / "a.pt"),
    }
    row.update(overrides)
    return row


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_raw_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _load_script(path: Path):
    name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_atomic_append_writes_fixed_columns_and_appends_new_keys(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    append_rows_atomic(output, [_row()])
    append_rows_atomic(
        output,
        [_row(model_or_arch_id="b", checkpoint_path=str((tmp_path / "b.pt").resolve()))],
    )

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_COLUMNS)
        assert [row["model_or_arch_id"] for row in reader] == ["a", "b"]


def test_atomic_append_rejects_batch_and_old_duplicates_without_changing_file(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    append_rows_atomic(output, [_row()])
    before = output.read_bytes()

    with pytest.raises(ValueError, match="duplicate"):
        append_rows_atomic(output, [_row(model_or_arch_id="b"), _row(model_or_arch_id="b")])
    assert output.read_bytes() == before

    with pytest.raises(ValueError, match="duplicate"):
        append_rows_atomic(output, [_row()])
    assert output.read_bytes() == before


def test_atomic_append_rejects_duplicate_keys_already_inside_old_csv(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    _write_raw_rows(output, [_row(), _row()])
    before = output.read_bytes()

    with pytest.raises(ValueError, match="duplicate.*existing|existing.*duplicate"):
        append_rows_atomic(output, [_row(model_or_arch_id="new")])

    assert output.read_bytes() == before


def test_evaluate_rejects_duplicate_keys_inside_old_csv_before_loader(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    _write_raw_rows(output, [_row(), _row()])
    before = output.read_bytes()
    loader_called = False

    def loader():
        nonlocal loader_called
        loader_called = True
        return object()

    with pytest.raises(ValueError, match="duplicate.*existing|existing.*duplicate"):
        evaluate_items(
            _items(tmp_path)[:1],
            lambda item: _Model(item.model_or_arch_id),
            loader,
            ["feature_delta"],
            output,
            _metadata(),
            compute_metric=lambda *_: 1.0,
            torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        )

    assert loader_called is False
    assert output.read_bytes() == before


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_append_failure_preserves_old_bytes_and_cleans_owned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    from robust_metric_lab import student_runner

    output = tmp_path / "metric_values.csv"
    append_rows_atomic(output, [_row()])
    before = output.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError(f"{operation} fixture")

    monkeypatch.setattr(student_runner.os, operation, fail)
    with pytest.raises(OSError, match=operation):
        append_rows_atomic(output, [_row(model_or_arch_id="b")])

    assert output.read_bytes() == before
    assert not output.with_name(output.name + ".tmp").exists()
    assert not output.with_name(output.name + ".lock").exists()


def test_atomic_append_refuses_existing_lock_without_deleting_it(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    lock = output.with_name(output.name + ".lock")
    lock.write_text("other process", encoding="utf-8")

    with pytest.raises(RuntimeError, match="lock"):
        append_rows_atomic(output, [_row()])

    assert lock.read_text(encoding="utf-8") == "other process"
    assert not output.exists()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda row: row.pop("runtime_sec"), "schema"),
        (lambda row: row.update(setting="nb201_e10"), "epoch"),
        (lambda row: row.update(seed=1), "seed"),
        (lambda row: row.update(direction="higher"), "direction"),
        (lambda row: row.update(metric_value=float("nan")), "finite"),
        (lambda row: row.update(target_value=""), "target"),
        (lambda row: row.update(runtime_sec=-1), "runtime"),
        (lambda row: row.update(benchmark="darts"), "benchmark"),
    ],
)
def test_append_validates_schema_protocol_direction_and_finite_values(
    tmp_path: Path, mutate, match: str
):
    row = _row()
    mutate(row)
    with pytest.raises(ValueError, match=match):
        append_rows_atomic(tmp_path / "metric_values.csv", [row])


def test_failed_row_may_have_empty_metric_but_requires_manifest_target(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    append_rows_atomic(
        output,
        [_row(status="failed", metric_value="", error="RuntimeError: broken fixture")],
    )
    assert _read(output)[0]["status"] == "failed"


class _Model:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class _Cuda:
    def __init__(self):
        self.empty_calls = 0

    def is_available(self):
        return True

    def empty_cache(self):
        self.empty_calls += 1


def _items(tmp_path: Path) -> list[BenchmarkManifestItem]:
    return [
        BenchmarkManifestItem(
            benchmark="nb201",
            model_or_arch_id=identifier,
            checkpoint_epoch=5,
            checkpoint_path=(tmp_path / f"{identifier}.pt").resolve(),
            target_value=0.5 + index / 10,
            metadata={"arch_string": f"arch-{identifier}"},
        )
        for index, identifier in enumerate(("one", "two", "three"))
    ]


def _metadata() -> dict[str, object]:
    return {"run_id": "run-1", "setting": "nb201_e5"}


def test_evaluate_preserves_item_metric_order_and_model_load_failures(tmp_path: Path):
    items = _items(tmp_path)
    models: list[_Model] = []
    loader_calls = 0
    frozen_samples = object()
    cuda = _Cuda()

    def loader():
        nonlocal loader_calls
        loader_calls += 1
        return frozen_samples

    def load_model(item):
        if item.model_or_arch_id == "two":
            raise RuntimeError("broken fixture")
        model = _Model(item.model_or_arch_id)
        models.append(model)
        return model

    def compute(metric, model, samples, item):
        assert samples is frozen_samples
        return 1.0 if metric == "feature_delta" else 2.0

    result = evaluate_items(
        items,
        load_model,
        loader,
        ["feature_delta", "logit_kl"],
        tmp_path / "metric_values.csv",
        _metadata(),
        compute_metric=compute,
        torch_module=SimpleNamespace(cuda=cuda),
    )

    rows = _read(tmp_path / "metric_values.csv")
    assert [(row["model_or_arch_id"], row["metric_name"]) for row in rows] == [
        (identifier, metric)
        for identifier in ("one", "two", "three")
        for metric in ("feature_delta", "logit_kl")
    ]
    assert [row["status"] for row in rows] == ["ok", "ok", "failed", "failed", "ok", "ok"]
    assert all("RuntimeError: broken fixture" in row["error"] for row in rows[2:4])
    assert loader_calls == 1
    assert [model.eval_calls for model in models] == [1, 1]
    assert cuda.empty_calls == 3
    assert result.written == 6 and result.failed == 2 and result.skipped == 0


def test_one_metric_failure_does_not_hide_other_metrics(tmp_path: Path):
    def compute(metric, *_args):
        if metric == "feature_delta":
            raise ArithmeticError("bad\nvalue")
        return 4.0

    evaluate_items(
        _items(tmp_path)[:1],
        lambda item: _Model(item.model_or_arch_id),
        object(),
        ["feature_delta", "logit_kl"],
        tmp_path / "metric_values.csv",
        _metadata(),
        compute_metric=compute,
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    rows = _read(tmp_path / "metric_values.csv")
    assert [row["status"] for row in rows] == ["failed", "ok"]
    assert rows[0]["error"] == "ArithmeticError: bad value"
    assert rows[1]["metric_value"] == "4.0"


@pytest.mark.parametrize("fatal", [KeyboardInterrupt, SystemExit])
def test_fatal_base_exceptions_propagate_after_prior_item_is_persisted(
    tmp_path: Path, fatal: type[BaseException]
):
    cuda = _Cuda()

    def compute(_metric, _model, _samples, item):
        if item.model_or_arch_id == "two":
            raise fatal()
        return 1.0

    with pytest.raises(fatal):
        evaluate_items(
            _items(tmp_path)[:2],
            lambda item: _Model(item.model_or_arch_id),
            object(),
            ["feature_delta"],
            tmp_path / "metric_values.csv",
            _metadata(),
            compute_metric=compute,
            torch_module=SimpleNamespace(cuda=cuda),
        )

    assert [row["model_or_arch_id"] for row in _read(tmp_path / "metric_values.csv")] == ["one"]
    assert cuda.empty_calls == 2


def test_resume_skips_ok_and_failed_exact_keys_without_loading_models(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    append_rows_atomic(output, [_row(model_or_arch_id="one")])
    append_rows_atomic(
        output,
        [
            _row(
                model_or_arch_id="two",
                status="failed",
                metric_value="",
                error="RuntimeError: old failure",
            )
        ],
    )

    def forbidden(_item):
        raise AssertionError("resume must not load fully completed items")

    result = evaluate_items(
        _items(tmp_path)[:2],
        forbidden,
        object(),
        ["feature_delta"],
        output,
        _metadata(),
        compute_metric=lambda *_: 1.0,
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    assert result.skipped == 2 and result.written == 0
    assert len(_read(output)) == 2


def test_resume_normalizes_rb40_none_epoch_before_pending_check(tmp_path: Path):
    output = tmp_path / "metric_values.csv"
    append_rows_atomic(
        output,
        [
            _row(
                benchmark="rb40",
                setting="rb40_mature",
                model_or_arch_id="rb40-id",
                checkpoint_epoch="",
                checkpoint_path=str((tmp_path / "rb40.pt").resolve()),
            )
        ],
    )

    def forbidden(_item):
        raise AssertionError("resume must recognize the normalized RB40 empty epoch")

    result = evaluate_items(
        [_single_item(tmp_path, "rb40")],
        forbidden,
        object(),
        ["feature_delta"],
        output,
        {"run_id": "run-1", "setting": "rb40_mature"},
        compute_metric=lambda *_: 1.0,
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert result.skipped == 1 and result.written == 0
    assert len(_read(output)) == 1


def _single_item(tmp_path: Path, benchmark: str) -> BenchmarkManifestItem:
    if benchmark == "rb40":
        epoch = None
    elif benchmark == "nb201":
        epoch = 5
    else:
        epoch = 100
    return BenchmarkManifestItem(
        benchmark,
        f"{benchmark}-id",
        epoch,
        (tmp_path / f"{benchmark}.pt").resolve(),
        0.5,
        {},
    )


def test_racl_dispatch_keeps_rb40_arch_and_proxy_semantics_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from robust_metric_lab import nasrobbench201

    calls = []

    def fixed(_model):
        calls.append("fixed")
        return SimpleNamespace(log_upper=3.0, mode="generic_chain_proxy")

    monkeypatch.setattr(nasrobbench201, "racl_fixed_network_lipschitz_stats", fixed)
    output = tmp_path / "rb.csv"
    evaluate_items(
        [_single_item(tmp_path, "rb40")],
        lambda _item: _Model("rb"),
        object(),
        ["racl_arch_lipschitz_log_upper", "racl_fixed_network_log_upper_proxy"],
        output,
        {"run_id": "rb-run", "setting": "rb40_mature"},
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    rows = _read(output)
    assert [row["status"] for row in rows] == ["failed", "ok"]
    assert "not applicable" in rows[0]["error"].lower()
    assert rows[1]["metric_value"] == "3.0"
    assert calls == ["fixed"]


def test_racl_dispatch_nb201_uses_arch_helper_and_rejects_fixed_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from robust_metric_lab import nasrobbench201

    calls = []

    def architecture(_model):
        calls.append("arch")
        return SimpleNamespace(log_upper=2.0, mode="nb201_cell")

    monkeypatch.setattr(nasrobbench201, "racl_architecture_lipschitz_stats", architecture)
    monkeypatch.setattr(
        nasrobbench201,
        "racl_fixed_network_lipschitz_stats",
        lambda _model: pytest.fail("NB201 must not use fixed-network proxy"),
    )
    output = tmp_path / "nb.csv"
    evaluate_items(
        [_single_item(tmp_path, "nb201")],
        lambda _item: _Model("nb"),
        object(),
        ["racl_arch_lipschitz_log_upper", "racl_fixed_network_log_upper_proxy"],
        output,
        {"run_id": "nb-run", "setting": "nb201_e5"},
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    rows = _read(output)
    assert [row["status"] for row in rows] == ["ok", "failed"]
    assert rows[0]["metric_value"] == "2.0"
    assert "not applicable" in rows[1]["error"].lower()
    assert calls == ["arch"]


def test_racl_dispatch_darts_requires_cell_mode_and_rejects_fixed_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from robust_metric_lab import nasrobbench201

    modes = iter(("generic_chain_proxy", "darts_cell"))

    def fixed(_model):
        return SimpleNamespace(log_upper=4.0, mode=next(modes))

    monkeypatch.setattr(nasrobbench201, "racl_fixed_network_lipschitz_stats", fixed)
    for run_id, expected_status in (("bad-mode", "failed"), ("cell-mode", "ok")):
        output = tmp_path / f"{run_id}.csv"
        evaluate_items(
            [_single_item(tmp_path, "darts")],
            lambda _item: _Model("darts"),
            object(),
            ["racl_arch_lipschitz_log_upper", "racl_fixed_network_log_upper_proxy"],
            output,
            {"run_id": run_id, "setting": "darts_e100"},
            torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        )
        rows = _read(output)
        assert rows[0]["status"] == expected_status
        assert rows[1]["status"] == "failed"
        if expected_status == "failed":
            assert "darts_cell" in rows[0]["error"]


def test_metric_helper_families_are_isolated_when_feature_bundle_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from robust_metric_lab import attacks, metrics, nasrobbench201, trnas

    monkeypatch.setattr(attacks, "make_adversarial", lambda *_args, **_kwargs: "adv")
    monkeypatch.setattr(
        metrics,
        "compute_feature_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("feature bundle failed")),
    )
    monkeypatch.setattr(
        metrics,
        "compute_feature_geometry_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(feature_hausdorff_delta=1.5),
    )
    monkeypatch.setattr(
        nasrobbench201,
        "racl_architecture_lipschitz_stats",
        lambda _model: SimpleNamespace(log_upper=2.5, mode="nb201_cell"),
    )
    monkeypatch.setattr(
        trnas,
        "compute_trnas_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            activation_psi=3.5, consistency_phi=4.5, r_score=5.5
        ),
    )
    output = tmp_path / "families.csv"
    evaluate_items(
        [_single_item(tmp_path, "nb201")],
        lambda _item: _Model("nb"),
        ("clean", "labels"),
        [
            "feature_delta",
            "feature_hausdorff_delta",
            "racl_arch_lipschitz_log_upper",
            "trnas_activation_psi",
        ],
        output,
        {"run_id": "families", "setting": "nb201_e5"},
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    rows = _read(output)
    assert [row["status"] for row in rows] == ["failed", "ok", "ok", "ok"]
    assert [row["metric_value"] for row in rows[1:]] == ["1.5", "2.5", "3.5"]


def test_cli_help_is_lazy_and_lists_required_flags_and_epoch_choices():
    common = (
        "--bundle-root",
        "--data-root",
        "--output-dir",
        "--run-id",
        "--sample-offset",
        "--num-samples",
        "--seed",
        "--item-limit",
        "--metric",
        "--device",
    )
    for script in SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert all(flag in completed.stdout for flag in common)
        if "nb201" in script.name:
            assert "{0,5,10,25,50}" in completed.stdout
        if "darts" in script.name:
            assert "{0,100}" in completed.stdout


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--run-id", "../escape"], "run-id"),
        (["--seed", "1"], "seed"),
        (["--sample-offset", "1"], "sample-offset"),
        (["--num-samples", "32"], "num-samples"),
        (["--metric", "not_registered"], "metric"),
    ],
)
def test_all_clis_reject_unsafe_or_nonfrozen_arguments(extra: list[str], match: str):
    base = [
        "--bundle-root", ".",
        "--data-root", ".",
        "--output-dir", ".",
        "--run-id", "safe",
        "--sample-offset", "0",
        "--num-samples", "64",
        "--seed", "20260808",
        "--item-limit", "1",
        "--metric", "feature_delta",
        "--device", "cpu",
    ]
    replaced = list(base)
    for flag, value in zip(extra[::2], extra[1::2]):
        index = replaced.index(flag)
        replaced[index + 1] = value
    for script in SCRIPTS:
        command = [sys.executable, str(script), *replaced]
        if "nb201" in script.name or "darts" in script.name:
            command.extend(["--epoch", "0"])
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 2
        assert match in completed.stderr.lower()


def test_rb40_adapter_uses_manifest_id_and_fixed_model_root(tmp_path: Path):
    module = _load_script(SCRIPTS[0])
    calls = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        return object()

    item = BenchmarkManifestItem(
        "rb40", "manifest-model", None, (tmp_path / "rb40/models/x.pt").resolve(), 0.4
    )
    result = module.make_model_loader(tmp_path, "cpu", load_rb_model_fn=fake_load)(item)
    assert result is not None
    assert calls == [{
        "model_name": "manifest-model",
        "dataset": "cifar10",
        "threat_model": "Linf",
        "device": "cpu",
        "model_dir": str((tmp_path / "rb40" / "models").resolve()),
    }]


class _CheckpointModel(_Model):
    def __init__(self):
        super().__init__("checkpoint")
        self.loaded = None
        self.device = None

    def load_state_dict(self, state, strict):
        self.loaded = (state, strict)

    def to(self, device):
        self.device = device
        return self


class _FakeTorch:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.load_calls = []

    def load(self, path, **kwargs):
        self.load_calls.append((Path(path), kwargs))
        assert Path(path) == self.checkpoint_path
        return {"model_state_dict": {"weight": 7}}


def test_nb201_adapter_uses_manifest_arch_and_strict_checkpoint(tmp_path: Path):
    module = _load_script(SCRIPTS[1])
    checkpoint = (tmp_path / "exact.pt").resolve()
    fake_torch = _FakeTorch(checkpoint)
    built = []
    model = _CheckpointModel()

    def build(arch_string, C, N):
        built.append((arch_string, C, N))
        return model

    arch = (
        "|skip_connect~0|+"
        "|nor_conv_1x1~0|skip_connect~1|+"
        "|avg_pool_3x3~0|nor_conv_3x3~1|none~2|"
    )
    item = BenchmarkManifestItem(
        "nb201", "17", 25, checkpoint, 0.8, {"arch_string": arch}
    )
    loaded = module.make_model_loader("cpu", build_model_fn=build, torch_module=fake_torch)(item)
    assert loaded is model
    assert built == [(arch, 16, 5)]
    assert model.loaded == ({"weight": 7}, True)
    assert model.device == "cpu"


def test_nb201_builder_uses_official_teacher_library_api(tmp_path: Path, monkeypatch):
    module = _load_script(SCRIPTS[1])
    library_root = tmp_path / "libnas" / "lib"
    library_root.mkdir(parents=True)
    calls = {}

    def dict2config(value, logger):
        calls["config"] = value
        calls["logger"] = logger
        return value

    class CellStructure:
        @staticmethod
        def str2structure(value):
            calls["arch"] = value
            return ("teacher-structure", value)

    def get_cell_based_tiny_net(config):
        import torch.nn as nn
        calls["network_config"] = config
        return nn.Identity()

    monkeypatch.setitem(sys.modules, "config_utils", SimpleNamespace(dict2config=dict2config))
    monkeypatch.setitem(
        sys.modules,
        "models",
        SimpleNamespace(
            CellStructure=CellStructure,
            get_cell_based_tiny_net=get_cell_based_tiny_net,
        ),
    )
    arch = (
        "|skip_connect~0|+"
        "|nor_conv_1x1~0|skip_connect~1|+"
        "|avg_pool_3x3~0|nor_conv_3x3~1|none~2|"
    )
    model = module.build_nb201_model(arch, C=16, N=5, library_root=library_root)
    assert calls["arch"] == arch
    assert calls["config"] == {
        "name": "infer.tiny",
        "C": 16,
        "N": 5,
        "genotype": ("teacher-structure", arch),
        "num_classes": 10,
    }
    assert len(model) == 2


def test_nb201_adapter_requires_standardized_manifest_arch_string(tmp_path: Path):
    module = _load_script(SCRIPTS[1])
    item = BenchmarkManifestItem(
        "nb201",
        "17",
        25,
        (tmp_path / "exact.pt").resolve(),
        0.8,
        {"genotype": {"arch_string": "legacy-ambiguous-value"}},
    )
    with pytest.raises(ValueError, match="arch_string"):
        module.make_model_loader(
            "cpu",
            build_model_fn=lambda *_args, **_kwargs: pytest.fail("must reject before build"),
            torch_module=_FakeTorch(item.checkpoint_path),
        )(item)


def test_nb201_manifest_loader_output_enters_adapter_with_fixed_macro(tmp_path: Path):
    module = _load_script(SCRIPTS[1])
    arch = (
        "|skip_connect~0|+"
        "|nor_conv_1x1~0|skip_connect~1|+"
        "|avg_pool_3x3~0|nor_conv_3x3~1|none~2|"
    )
    manifest = tmp_path / "nb201.csv"
    rows = [
        {
            "arch_id": "17",
            "epoch": epoch,
            "checkpoint_path": f"checkpoints/17-{epoch}.pt",
            "official_final_robust_accuracy": "0.8",
            "genotype_json": json.dumps({"arch_string": arch}),
            "metadata_json": json.dumps({"source": "teacher"}),
        }
        for epoch in (0, 5, 10, 25, 50)
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    item = load_nb201_items(manifest, tmp_path)[3]
    fake_torch = _FakeTorch(item.checkpoint_path)
    built = []
    model = _CheckpointModel()

    def build(arch_string, C, N):
        built.append((arch_string, C, N))
        return model

    module.make_model_loader(
        "cpu", build_model_fn=build, torch_module=fake_torch
    )(item)
    assert built == [(arch, 16, 5)]


def test_darts_adapter_uses_manifest_genotype_and_fixed_macro(tmp_path: Path):
    module = _load_script(SCRIPTS[2])
    checkpoint = (tmp_path / "exact.pt").resolve()
    fake_torch = _FakeTorch(checkpoint)
    calls = []
    model = _CheckpointModel()
    genotype = object()

    class FakeGenotype:
        @classmethod
        def from_dict(cls, value):
            calls.append(("genotype", value))
            return genotype

    def build_network(parsed, **kwargs):
        calls.append(("network", parsed, kwargs))
        return object()

    def wrap(network):
        calls.append(("wrap", network))
        return model

    item = BenchmarkManifestItem(
        "darts", "d7", 100, checkpoint, 0.6, {"genotype": {"normal": []}}
    )
    loaded = module.make_model_loader(
        "cpu",
        genotype_cls=FakeGenotype,
        network_cls=build_network,
        wrap_model_fn=wrap,
        torch_module=fake_torch,
    )(item)
    assert loaded is model
    assert calls[0] == ("genotype", {"normal": []})
    assert calls[1] == (
        "network",
        genotype,
        {"channels": 16, "layers": 8, "num_classes": 10},
    )
    assert calls[2][0] == "wrap"
    assert model.loaded == ({"weight": 7}, True)
    assert model.device == "cpu"
