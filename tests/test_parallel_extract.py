"""Chunking, archive part discovery and the optional local archive copy (no 7z needed)."""
from src.archive_utils import _chunks, archive_part_files, copy_archive_local, copy_file_timed


def test_chunks():
    items = list(range(10))
    c = _chunks(items, n_chunks=3, max_size=1000)
    assert [len(x) for x in c] == [4, 4, 2] and sum(c, []) == items
    c = _chunks(items, n_chunks=2, max_size=3)          # max_size wins
    assert [len(x) for x in c] == [3, 3, 3, 1]
    assert _chunks([], 4, 10) == []


def _fake_archive(d):
    d.mkdir()
    for name, size in [("arc.z02", 20), ("arc.z01", 10), ("arc.z10", 5), ("arc.zip", 3), ("other.z01", 7)]:
        (d / name).write_bytes(b"x" * size)
    return d / "arc.zip"


def test_archive_part_files_sorted(tmp_path):
    z = _fake_archive(tmp_path / "gen")
    assert [p.name for p in archive_part_files(z)] == ["arc.z01", "arc.z02", "arc.z10", "arc.zip"]


def test_copy_archive_local_and_resume(tmp_path):
    z = _fake_archive(tmp_path / "gen")
    dest = tmp_path / "local"
    local_zip = copy_archive_local(z, dest, margin_gb=0, logger=None)
    assert local_zip == dest / "arc.zip"
    assert sorted(p.name for p in dest.iterdir()) == ["arc.z01", "arc.z02", "arc.z10", "arc.zip"]
    assert (dest / "arc.z02").stat().st_size == 20
    # resume: a part with the wrong size gets copied again, others are skipped
    (dest / "arc.z02").write_bytes(b"x")
    mtime = (dest / "arc.z01").stat().st_mtime_ns
    copy_archive_local(z, dest, margin_gb=0)
    assert (dest / "arc.z02").stat().st_size == 20
    assert (dest / "arc.z01").stat().st_mtime_ns == mtime


def test_copy_archive_local_falls_back_when_disk_too_small(tmp_path):
    z = _fake_archive(tmp_path / "gen")
    assert copy_archive_local(z, tmp_path / "local", margin_gb=1e9) is None


def test_copy_file_timed(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"y" * 1000)
    mbps = copy_file_timed(src, tmp_path / "out" / "a.bin")
    assert mbps > 0 and (tmp_path / "out" / "a.bin").read_bytes() == src.read_bytes()
