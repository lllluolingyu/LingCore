import os
import stat

import pytest

from lingcore.evaluation.storage import read_bytes, read_json, write_json
from lingcore.paths import PathEscapeError, ReadLimitError


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


@pytest.mark.parametrize("grows_during_read", [False, True])
def test_bounded_read_identifies_size_limit(tmp_path, monkeypatch, grows_during_read):
    path = tmp_path / "artifact"
    path.write_bytes(b"x" if grows_during_read else b"x" * 9)
    if grows_during_read:
        fdopen = os.fdopen

        def grow_then_open(fd, *args, **kwargs):
            with path.open("ab") as stream:
                stream.write(b"x" * 8)
            return fdopen(fd, *args, **kwargs)

        monkeypatch.setattr(os, "fdopen", grow_then_open)
    with pytest.raises(ReadLimitError, match="read limit"):
        read_bytes(path, max_bytes=8)


def test_parent_locators_are_normalized_without_following_symlinks(tmp_path):
    parent = tmp_path / "child"
    parent.mkdir()
    path = parent / ".." / "records" / "record.json"
    write_json(path, {"ok": True})
    assert read_json(path) == {"ok": True}
    (tmp_path / "link").symlink_to(tmp_path / "records", target_is_directory=True)
    with pytest.raises(PathEscapeError):
        read_json(parent / ".." / "link" / "record.json")
