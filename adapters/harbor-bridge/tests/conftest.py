from pathlib import Path

import pytest


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    task = root / "tiny"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment/Dockerfile").write_text("FROM debian:bookworm-slim\n")
    (task / "tests/test.sh").write_text("#!/bin/bash\nexit 0\n")
    (task / "instruction.md").write_text("<!-- synthetic canary -->\n\nWrite a file.\n")
    (task / "task.toml").write_text("""schema_version = "2.0"
[task]
name = "fixture/tiny"
version = "1.0.0"
[agent]
timeout_sec = 120
[environment]
cpus = 2
memory_mb = 4096
storage_mb = 10240
""")
    return root
