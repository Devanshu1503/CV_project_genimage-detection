"""Tests for parsing `7z l -slt` output and discovering split-zip parts (no 7z needed)."""
import pytest

from src.archive_utils import archive_inventory, discover_archives, parse_7z_slt

# Hand-written sample in the format printed by p7zip 16.02 (`7z l -slt`).
SAMPLE_SLT = """
7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=C.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs)

Scanning the drive for archives:
1 file, 1234567 bytes (1206 KiB)

Listing archive: imagenet_ai_0419_sdv4.zip

--
Path = imagenet_ai_0419_sdv4.zip
Type = zip
Physical Size = 1234567
Total Physical Size = 90000000000
Multivolume = +
Volume Index = 29
Volumes = 30

----------
Path = imagenet_ai_0419_sdv4/train/ai
Folder = +
Size = 0
Packed Size = 0
Modified = 2023-04-19 10:00:00
Created =
Accessed =
Attributes = D drwxr-xr-x
Encrypted = -
Comment =
CRC =
Method = Store
Host OS = Unix
Version = 10
Volume Index = 0

Path = imagenet_ai_0419_sdv4/train/ai/001_sdv4_00001.png
Folder = -
Size = 412345
Packed Size = 410000
Modified = 2023-04-19 10:00:01
Created =
Accessed =
Attributes = A -rw-r--r--
Encrypted = -
Comment =
CRC = 1A2B3C4D
Method = Deflate
Host OS = Unix
Version = 20
Volume Index = 0
Offset = 120

Path = imagenet_ai_0419_sdv4/train/nature/n01440764_10026.JPEG
Folder = -
Size = 110034
Packed Size = 109000
Modified = 2023-04-19 10:00:02
Attributes = A -rw-r--r--
CRC = 5E6F7A8B
Method = Store
Volume Index = 3

Path = imagenet_ai_0419_sdv4/val/nature/ILSVRC2012_val_00000001.JPEG
Folder = -
Size = 98765
Packed Size = 98000
Modified = 2023-04-19 10:00:03
Attributes = A -rw-r--r--
CRC = 00000000
Method = Store
Volume Index = 29
"""


def test_parse_7z_slt_entries():
    df = parse_7z_slt(SAMPLE_SLT)
    # The archive-level block (Path = imagenet_ai_0419_sdv4.zip) must NOT be an entry.
    assert len(df) == 4
    assert "imagenet_ai_0419_sdv4.zip" not in df["inner_path"].tolist()
    assert df["is_dir"].tolist() == [True, False, False, False]
    files = df.loc[~df["is_dir"]]
    assert files["size"].tolist() == [412345, 110034, 98765]
    assert files["crc"].iloc[0] == "1A2B3C4D"
    assert files["inner_path"].iloc[1] == "imagenet_ai_0419_sdv4/train/nature/n01440764_10026.JPEG"


def test_parse_7z_slt_windows_backslashes():
    text = "----------\nPath = top\\train\\ai\\a.png\nFolder = -\nSize = 5\n"
    df = parse_7z_slt(text)
    assert df["inner_path"].iloc[0] == "top/train/ai/a.png"


def test_parse_7z_slt_rejects_garbage():
    with pytest.raises(ValueError):
        parse_7z_slt("ERROR: Can not open the file as archive")


def test_discover_archives(tmp_path):
    gen = tmp_path / "Midjourney"
    gen.mkdir()
    for name in ["imagenet_midjourney.z01", "imagenet_midjourney.z02", "imagenet_midjourney.z04",
                 "imagenet_midjourney.zip", "README.txt"]:
        (gen / name).write_bytes(b"x" * 10)
    arcs = discover_archives(gen)
    assert len(arcs) == 1
    a = arcs[0]
    assert a["base"] == "imagenet_midjourney"
    assert a["has_zip"] and a["n_parts"] == 3
    assert a["missing_parts"] == [3]          # .z03 is missing
    assert a["total_bytes"] == 40             # 3 parts + .zip, README ignored

    inv = archive_inventory(tmp_path, ["Midjourney", "BigGAN"])
    assert inv.loc[inv.generator == "BigGAN", "status"].iloc[0] == "FOLDER NOT FOUND"
    assert "MISSING parts" in inv.loc[inv.generator == "Midjourney", "status"].iloc[0]
