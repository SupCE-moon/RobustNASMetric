import hashlib

import pytest

from scripts.activation_survival_manifest import verify_source_inventory


def test_audited_source_rejects_mutation(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"trusted")
    inventory = {"runner.py": hashlib.sha256(b"trusted").hexdigest()}
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_source_inventory(inventory, tmp_path)


def test_audited_source_rejects_missing_file(tmp_path):
    inventory = {"runner.py": hashlib.sha256(b"trusted").hexdigest()}
    with pytest.raises(ValueError, match="source mismatch"):
        verify_source_inventory(inventory, tmp_path)
