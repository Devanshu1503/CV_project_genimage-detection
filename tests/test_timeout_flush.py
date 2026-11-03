"""The config timeout reaches subprocess.run, and flush_drive is a no-op locally."""
import subprocess

import pandas as pd

from src import archive_utils as au
from src.colab_setup import flush_drive
from src.config import load_config


def test_config_timeout_reaches_7z(tmp_path, monkeypatch):
    timeout = load_config()["seven_zip"]["extract_timeout_sec"]
    assert timeout == 2400
    seen = []

    def fake_run(cmd, **kw):
        seen.append(kw.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(au, "find_7z", lambda b="auto": "7z")
    monkeypatch.setattr(au.subprocess, "run", fake_run)
    sample = pd.DataFrame({"sample_id": ["a", "b"], "inner_path": ["x/a.png", "x/b.png"],
                           "raw_relpath": ["raw/a.png", "raw/b.png"], "uncompressed_bytes": [1, 1]})
    au.extract_sample_to_drive(sample, "arc.zip", tmp_path / "w", tmp_path / "d", batch_size=1,
                               timeout=timeout, progress=False, workers=2)
    assert seen == [timeout, timeout]


def test_flush_drive_local(capsys):
    flush_drive()
    assert "nothing to flush" in capsys.readouterr().out
