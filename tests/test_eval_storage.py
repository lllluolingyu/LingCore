import os
import stat

import pytest

from lingcore.evaluation.storage import read_json, write_json


@pytest.mark.parametrize("replace", [False, True])
def test_records_sync_bytes_and_directory_entries(tmp_path, monkeypatch, replace):
    path = tmp_path / "nested/job/record.json"
    if replace:
        write_json(path, {"old": True})
    syncs = []
    original = os.fsync

    def sync(fd):
        info = os.fstat(fd)
        if stat.S_ISREG(info.st_mode):
            syncs.append("file")
        elif stat.S_ISDIR(info.st_mode):
            syncs.append("directory")
        original(fd)

    monkeypatch.setattr(os, "fsync", sync)
    write_json(path, {"durable": True}, replace=replace)
    assert read_json(path) == {"durable": True}
    assert syncs[-2:] == ["file", "directory"]
    assert path.stat().st_mode & 0o777 == 0o600
