"""Read GenImage split-zip archives IN PLACE with 7-Zip.

GenImage ships every generator as a split ZIP: <base>.z01 ... <base>.zNN plus a
final <base>.zip that holds the central directory. We never copy or extract a
whole generator (tens of GB). Instead we:
  1. `7z l -slt <base>.zip`   -> list every file inside (technical listing)
  2. parse that text into a DataFrame (parse_7z_slt)
  3. `7z x <base>.zip -i@list` -> extract only the files we sampled

Archive base names differ per generator, so they are discovered, never hardcoded.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# Matches "<base>.zip" or "<base>.z01", "<base>.z123" (case-insensitive).
_PART_RE = re.compile(r"^(?P<base>.+)\.(?P<ext>zip|z\d{2,})$", re.IGNORECASE)


class SevenZipError(RuntimeError):
    """Raised when a 7z command fails. Carries the exact command and stderr."""

    def __init__(self, cmd, returncode, stdout, stderr):
        self.cmd, self.returncode, self.stdout, self.stderr = cmd, returncode, stdout, stderr
        msg = (f"7z failed (exit code {returncode}).\n  command: {' '.join(map(str, cmd))}\n"
               f"  stderr: {stderr.strip()[-2000:]}\n  stdout tail: {stdout.strip()[-1000:]}")
        super().__init__(msg)


# -----------------------------------------------------------------------------
# 7z binary
# -----------------------------------------------------------------------------
def find_7z(binary: str = "auto") -> str:
    """Return the 7-Zip executable to use, or raise with an install hint."""
    if binary and binary != "auto":
        if shutil.which(binary) or Path(binary).exists():
            return binary
        raise FileNotFoundError(f"Configured 7z binary not found: {binary}")
    for cand in ("7z", "7zz", "7za"):
        if shutil.which(cand):
            return cand
    raise FileNotFoundError(
        "No 7-Zip binary found. In Colab run: !apt-get install -y p7zip-full "
        "(colab_setup.setup() does this automatically).")


def seven_zip_version(binary: str = "auto") -> str:
    """First non-empty line of 7z's banner, e.g. '7-Zip [64] 16.02 : ...'."""
    exe = find_7z(binary)
    res = subprocess.run([exe], capture_output=True, text=True)
    for line in (res.stdout + res.stderr).splitlines():
        if line.strip():
            return f"{exe}: {line.strip()}"
    return f"{exe}: (no version banner)"


# -----------------------------------------------------------------------------
# Archive discovery
# -----------------------------------------------------------------------------
def discover_archives(gen_dir: str | Path, max_depth: int = 2) -> list[dict]:
    """Find every split-zip set in a generator folder (searching a few levels down).

    Returns one dict per archive base name with its parts, sizes and any gaps
    in the .z01..zNN numbering (a gap means a part is missing on Drive).
    """
    gen_dir = Path(gen_dir)
    groups: dict[tuple[Path, str], dict] = {}
    for f in _walk_files(gen_dir, max_depth):
        m = _PART_RE.match(f.name)
        if not m:
            continue
        key = (f.parent, m.group("base"))
        g = groups.setdefault(key, {"folder": str(f.parent), "base": m.group("base"),
                                    "zip_path": None, "parts": [], "total_bytes": 0})
        size = f.stat().st_size
        g["total_bytes"] += size
        if m.group("ext").lower() == "zip":
            g["zip_path"] = str(f)
        else:
            g["parts"].append((int(m.group("ext")[1:]), str(f), size))
    out = []
    for g in groups.values():
        nums = sorted(n for n, _, _ in g["parts"])
        expected = list(range(1, (max(nums) if nums else 0) + 1))
        g["parts"] = [p for _, p, _ in sorted(g["parts"])]
        g["n_parts"] = len(nums)
        g["missing_parts"] = sorted(set(expected) - set(nums))
        g["has_zip"] = g["zip_path"] is not None
        out.append(g)
    return sorted(out, key=lambda d: d["base"])


def _walk_files(root: Path, max_depth: int) -> Iterable[Path]:
    if not root.exists():
        return
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        for p in sorted(d.iterdir()):
            if p.is_file():
                yield p
            elif p.is_dir() and depth < max_depth:
                stack.append((p, depth + 1))


def archive_inventory(genimage_root: str | Path, generator_folders: list[str]) -> pd.DataFrame:
    """One row per (generator, archive) with part counts and total size in GB."""
    rows = []
    for gen in generator_folders:
        gdir = Path(genimage_root) / gen
        if not gdir.exists():
            rows.append({"generator": gen, "status": "FOLDER NOT FOUND"})
            continue
        archives = discover_archives(gdir)
        if not archives:
            all_files = [p.name for p in _walk_files(gdir, 2)]
            rows.append({"generator": gen, "status": f"NO ZIP FOUND ({len(all_files)} files)"})
            continue
        for a in archives:
            status = "ok"
            if not a["has_zip"]:
                status = "MISSING final .zip"
            elif a["missing_parts"]:
                status = f"MISSING parts {a['missing_parts']}"
            rows.append({"generator": gen, "base": a["base"], "zip_path": a["zip_path"],
                         "n_parts": a["n_parts"], "has_zip": a["has_zip"],
                         "total_gb": round(a["total_bytes"] / 1e9, 2), "status": status})
    return pd.DataFrame(rows)


def list_folder_files(gen_dir: str | Path) -> pd.DataFrame:
    """Plain file listing (name, size in GB) of one generator folder."""
    return pd.DataFrame([{"file": str(p.relative_to(gen_dir)), "gb": round(p.stat().st_size / 1e9, 3)}
                         for p in _walk_files(Path(gen_dir), 2)])


# -----------------------------------------------------------------------------
# Listing
# -----------------------------------------------------------------------------
def run_7z_list(zip_path: str | Path, binary: str = "auto", timeout: int | None = None) -> str:
    """Run `7z l -slt` on the archive and return stdout. Raises SevenZipError on failure."""
    exe = find_7z(binary)
    cmd = [exe, "l", "-slt", str(zip_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         encoding="utf-8", errors="replace")
    # Exit code 0 = OK, 1 = warning (listing is still usable), >=2 = fatal error.
    if res.returncode >= 2 or "----------" not in res.stdout:
        raise SevenZipError(cmd, res.returncode, res.stdout, res.stderr)
    if res.returncode == 1:
        print(f"[7z] WARNING while listing {zip_path}: {res.stderr.strip()[-500:]}")
    return res.stdout


def parse_7z_slt(text: str) -> pd.DataFrame:
    """Parse `7z l -slt` output into one row per archive entry.

    The output looks like:
        ...header...
        --
        Path = archive.zip        <- archive-level block (ignored)
        Type = zip
        ----------                <- entries start after this line
        Path = top/train/ai/x.png
        Folder = -
        Size = 12345
        Packed Size = 12000
        Modified = 2023-04-19 10:00:00
        Attributes = A
        CRC = 1A2B3C4D
        ...
        (blank line between entries)
    """
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "----------") + 1
    except StopIteration:
        raise ValueError("No '----------' separator found: not a valid `7z l -slt` output.")
    records, cur = [], None
    for ln in lines[start:]:
        if ln.startswith("Path = "):
            if cur is not None:
                records.append(cur)
            cur = {"Path": ln[len("Path = "):]}
        elif cur is not None and " = " in ln:
            k, v = ln.split(" = ", 1)
            cur[k.strip()] = v.strip()
        elif cur is not None and ln.endswith(" =") and not ln.startswith(" "):
            cur[ln[:-2].strip()] = ""      # key with empty value, e.g. "CRC ="
    if cur is not None:
        records.append(cur)

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["inner_path", "is_dir", "size", "packed_size", "modified", "crc"])
    def col(name, default=None):
        # Not every 7z version prints every field; missing ones become a default.
        return df[name] if name in df else pd.Series(default, index=df.index, dtype=object)

    out = pd.DataFrame({"inner_path": df["Path"].str.replace("\\", "/", regex=False)})
    folder = col("Folder", "-").fillna("-")
    attrs = col("Attributes", "").fillna("")
    out["is_dir"] = (folder == "+") | attrs.str.startswith("D")
    out["size"] = pd.to_numeric(col("Size"), errors="coerce").astype("Int64")
    out["packed_size"] = pd.to_numeric(col("Packed Size"), errors="coerce").astype("Int64")
    out["modified"] = col("Modified")
    out["crc"] = col("CRC")
    return out


def list_archive(zip_path: str | Path, binary: str = "auto", timeout: int | None = None,
                 cache_txt: str | Path | None = None) -> tuple[pd.DataFrame, float]:
    """List an archive. Returns (entries DataFrame, seconds taken by 7z).

    If cache_txt is given, the raw 7z text is saved there (and re-used when it
    already exists) so parsing can be re-run without touching Drive again.
    """
    t0 = time.time()
    if cache_txt and Path(cache_txt).exists():
        text = Path(cache_txt).read_text(encoding="utf-8")
    else:
        text = run_7z_list(zip_path, binary, timeout)
        if cache_txt:
            Path(cache_txt).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_txt).write_text(text, encoding="utf-8")
    secs = time.time() - t0
    return parse_7z_slt(text), secs


# -----------------------------------------------------------------------------
# Selective extraction
# -----------------------------------------------------------------------------
def extract_files(zip_path: str | Path, inner_paths: list[str], out_dir: str | Path,
                  binary: str = "auto", timeout: int | None = None,
                  list_file: str | Path | None = None) -> dict:
    """Extract ONLY `inner_paths` from the archive into out_dir (keeping the inner folders).

    Uses `7z x <zip> -o<out_dir> -i@<listfile> -y`. Returns a dict with timing,
    which requested files exist afterwards, and which are missing.
    """
    exe = find_7z(binary)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    list_file = Path(list_file) if list_file else out_dir / "_filelist.txt"
    list_file.write_text("\n".join(inner_paths) + "\n", encoding="utf-8")
    cmd = [exe, "x", str(zip_path), f"-o{out_dir}", f"-i@{list_file}", "-y", "-scsUTF-8", "-bd"]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         encoding="utf-8", errors="replace")
    secs = time.time() - t0
    if res.returncode >= 2:
        raise SevenZipError(cmd, res.returncode, res.stdout, res.stderr)
    found, missing = [], []
    for p in inner_paths:
        (found if (out_dir / p).exists() else missing).append(p)
    return {"seconds": secs, "found": found, "missing": missing,
            "returncode": res.returncode, "stderr": res.stderr.strip()}


def extract_sample_to_drive(sample: pd.DataFrame, zip_path: str | Path, work_dir: str | Path,
                            dest_root: str | Path, batch_size: int = 1000, binary: str = "auto",
                            timeout: int | None = None,
                            on_file: Callable[[Path, pd.Series], dict] | None = None,
                            logger=None, progress: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Extract the sampled files of ONE generator and copy them to Drive.

    `sample` needs columns: sample_id, inner_path, raw_relpath, uncompressed_bytes.
    Files whose Drive copy already exists with the right size are skipped (resumable).
    Original bytes are copied unchanged (no re-encoding).
    `on_file(local_path, row)` is called on the LOCAL copy before it is deleted
    (fast disk) and can return a dict of extra fields (e.g. image metadata).

    Returns (status DataFrame: sample_id, status, ...; list of on_file dicts).
    """
    from tqdm.auto import tqdm

    log = logger.info if logger else print
    dest_root, work_dir = Path(dest_root), Path(work_dir)
    rows = sample.copy()
    rows["dest"] = rows["raw_relpath"].map(lambda r: dest_root / r)
    done_mask = rows.apply(lambda r: r["dest"].exists() and
                           r["dest"].stat().st_size == int(r["uncompressed_bytes"]), axis=1)
    status = [{"sample_id": s, "status": "already_done"} for s in rows.loc[done_mask, "sample_id"]]
    todo = rows.loc[~done_mask]
    log(f"  {len(rows)} files sampled | {done_mask.sum()} already on Drive | {len(todo)} to extract")
    extra: list[dict] = []
    if todo.empty:
        return pd.DataFrame(status), extra

    batches = [todo.iloc[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    for bi, batch in enumerate(tqdm(batches, desc="7z batches", disable=not progress)):
        tmp = work_dir / f"_extract_batch"
        if tmp.exists():
            shutil.rmtree(tmp)
        res = extract_files(zip_path, batch["inner_path"].tolist(), tmp, binary, timeout)
        mb = batch["uncompressed_bytes"].sum() / 1e6
        log(f"  batch {bi + 1}/{len(batches)}: {len(res['found'])}/{len(batch)} files, "
            f"{mb:.0f} MB in {res['seconds']:.0f}s ({mb / max(res['seconds'], 1e-6):.1f} MB/s)")
        if res["returncode"] == 1:
            log(f"  7z WARNING: {res['stderr'][-500:]}")
        for _, r in batch.iterrows():
            src = tmp / r["inner_path"]
            if not src.exists():
                status.append({"sample_id": r["sample_id"], "status": "missing_after_extract"})
                continue
            if on_file is not None:
                try:
                    info = on_file(src, r) or {}
                    info["sample_id"] = r["sample_id"]
                    extra.append(info)
                except Exception as e:  # never silently skip: record the error
                    extra.append({"sample_id": r["sample_id"], "on_file_error": repr(e)})
            dest = r["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            shutil.copyfile(src, part)          # copy bytes unchanged
            part.replace(dest)                  # atomic rename -> no half-written files
            status.append({"sample_id": r["sample_id"], "status": "extracted"})
        shutil.rmtree(tmp, ignore_errors=True)
    return pd.DataFrame(status), extra
