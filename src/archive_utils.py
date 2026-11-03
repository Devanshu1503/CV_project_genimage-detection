"""Read the GenImage split zips in place with 7z (list + extract only what we need)."""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# <base>.zip or <base>.z01, .z02, ...
_PART_RE = re.compile(r"^(?P<base>.+)\.(?P<ext>zip|z\d{2,})$", re.IGNORECASE)


class SevenZipError(RuntimeError):
    """7z failed (keeps the command and stderr)."""

    def __init__(self, cmd, returncode, stdout, stderr):
        self.cmd, self.returncode, self.stdout, self.stderr = cmd, returncode, stdout, stderr
        msg = (f"7z failed (exit code {returncode}).\n  command: {' '.join(map(str, cmd))}\n"
               f"  stderr: {stderr.strip()[-2000:]}\n  stdout tail: {stdout.strip()[-1000:]}")
        super().__init__(msg)


# 7z binary
def find_7z(binary: str = "auto") -> str:
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
    """7z version line."""
    exe = find_7z(binary)
    res = subprocess.run([exe], capture_output=True, text=True)
    for line in (res.stdout + res.stderr).splitlines():
        if line.strip():
            return f"{exe}: {line.strip()}"
    return f"{exe}: (no version banner)"


# Archive discovery
def discover_archives(gen_dir: str | Path, max_depth: int = 2) -> list[dict]:
    """Find the split-zip sets in a generator folder."""
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
    """Parts, size and status per generator."""
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
    """Files + sizes in one generator folder."""
    return pd.DataFrame([{"file": str(p.relative_to(gen_dir)), "gb": round(p.stat().st_size / 1e9, 3)}
                         for p in _walk_files(Path(gen_dir), 2)])


# Listing
def run_7z_list(zip_path: str | Path, binary: str = "auto", timeout: int | None = None) -> str:
    """Run `7z l -slt` and return the raw text."""
    exe = find_7z(binary)
    cmd = [exe, "l", "-slt", str(zip_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         encoding="utf-8", errors="replace")
    # 0 ok, 1 warning, 2+ fatal
    if res.returncode >= 2 or "----------" not in res.stdout:
        raise SevenZipError(cmd, res.returncode, res.stdout, res.stderr)
    if res.returncode == 1:
        print(f"[7z] WARNING while listing {zip_path}: {res.stderr.strip()[-500:]}")
    return res.stdout


def parse_7z_slt(text: str) -> pd.DataFrame:
    """Parse `7z l -slt` text into a DataFrame (one row per entry)."""
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
            cur[ln[:-2].strip()] = ""  # e.g. "CRC ="
    if cur is not None:
        records.append(cur)

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["inner_path", "is_dir", "size", "packed_size", "modified", "crc"])
    def col(name, default=None):
        # some 7z versions skip fields
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
    """List an archive, returns (entries, seconds)."""
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


# Selective extraction
def extract_files(zip_path: str | Path, inner_paths: list[str], out_dir: str | Path,
                  binary: str = "auto", timeout: int | None = None,
                  list_file: str | Path | None = None) -> dict:
    """Extract only `inner_paths` into out_dir."""
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


def _chunks(items: list, n_chunks: int, max_size: int) -> list[list]:
    """Split into contiguous chunks, at least n_chunks, each <= max_size."""
    if not items:
        return []
    size = max(1, min(max_size, -(-len(items) // max(n_chunks, 1))))
    return [items[i:i + size] for i in range(0, len(items), size)]


def parallel_extract(zip_path: str | Path, inner_paths: list[str], out_dir: str | Path, workers: int = 4,
                     binary: str = "auto", timeout: int | None = None) -> dict:
    """Extract with several 7z processes at once (one temp dir each), used by the benchmark."""
    from concurrent.futures import ThreadPoolExecutor
    out_dir = Path(out_dir)
    chunks = _chunks(sorted(inner_paths), workers, len(inner_paths))
    t0 = time.time()
    with ThreadPoolExecutor(max(workers, 1)) as ex:
        results = list(ex.map(lambda kc: extract_files(zip_path, kc[1], out_dir / f"w{kc[0]}", binary, timeout),
                              enumerate(chunks)))
    found = [f for r in results for f in r["found"]]
    missing = [f for r in results for f in r["missing"]]
    return {"seconds": time.time() - t0, "found": found, "missing": missing}


def extract_sample_to_drive(sample: pd.DataFrame, zip_path: str | Path, work_dir: str | Path,
                            dest_root: str | Path, batch_size: int = 1000, binary: str = "auto",
                            timeout: int | None = None,
                            on_file: Callable[[Path, pd.Series], dict] | None = None,
                            logger=None, progress: bool = True, workers: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    """Extract one generator's sample and copy it to Drive (skips files already there).

    Files are sorted by inner_path and chunked, `workers` 7z processes run at once,
    each into its own temp dir.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    log = logger.info if logger else print
    dest_root, work_dir = Path(dest_root), Path(work_dir)
    rows = sample.copy()
    rows["dest"] = rows["raw_relpath"].map(lambda r: dest_root / r)
    done_mask = rows.apply(lambda r: r["dest"].exists() and
                           r["dest"].stat().st_size == int(r["uncompressed_bytes"]), axis=1)
    status = [{"sample_id": s, "status": "already_done"} for s in rows.loc[done_mask, "sample_id"]]
    todo = rows.loc[~done_mask].sort_values("inner_path")
    log(f"  {len(rows)} files sampled | {done_mask.sum()} already on Drive | {len(todo)} to extract "
        f"| {workers} worker(s)")
    extra: list[dict] = []
    if todo.empty:
        return pd.DataFrame(status), extra

    lock = threading.Lock()
    batches = [todo.iloc[idx] for idx in _chunks(list(range(len(todo))), workers, batch_size)]

    def run_batch(bi, batch):
        tmp = work_dir / f"_extract_batch_{bi}"
        if tmp.exists():
            shutil.rmtree(tmp)
        res = extract_files(zip_path, batch["inner_path"].tolist(), tmp, binary, timeout)
        mb = batch["uncompressed_bytes"].sum() / 1e6
        log(f"  batch {bi + 1}/{len(batches)}: {len(res['found'])}/{len(batch)} files, "
            f"{mb:.0f} MB in {res['seconds']:.0f}s ({mb / max(res['seconds'], 1e-6):.1f} MB/s)")
        if res["returncode"] == 1:
            log(f"  7z WARNING: {res['stderr'][-500:]}")
        st, ex_rows = [], []
        for _, r in batch.iterrows():
            src = tmp / r["inner_path"]
            if not src.exists():
                st.append({"sample_id": r["sample_id"], "status": "missing_after_extract"})
                continue
            if on_file is not None:
                try:
                    info = on_file(src, r) or {}
                    info["sample_id"] = r["sample_id"]
                    ex_rows.append(info)
                except Exception as e:  # log it, don't skip silently
                    ex_rows.append({"sample_id": r["sample_id"], "on_file_error": repr(e)})
            dest = r["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            shutil.copyfile(src, part)  # original bytes
            part.replace(dest)  # no half-written files
            st.append({"sample_id": r["sample_id"], "status": "extracted"})
        shutil.rmtree(tmp, ignore_errors=True)
        with lock:
            status.extend(st)
            extra.extend(ex_rows)

    with ThreadPoolExecutor(max(workers, 1)) as ex:
        futs = [ex.submit(run_batch, bi, b) for bi, b in enumerate(batches)]
        for f in tqdm(as_completed(futs), total=len(futs), desc="7z batches", disable=not progress):
            f.result()  # re-raise worker errors
    return pd.DataFrame(status), extra


# Optional: copy a whole archive to local disk first (faster than reading Drive)
def archive_part_files(zip_path: str | Path) -> list[Path]:
    """All .zNN parts plus the .zip of one archive."""
    zip_path = Path(zip_path)
    base = zip_path.name[:-4]
    parts = []
    for p in zip_path.parent.iterdir():
        m = _PART_RE.match(p.name)
        if m and m.group("base") == base and p != zip_path:
            parts.append(p)
    return sorted(parts, key=lambda p: int(p.suffix[2:])) + [zip_path]


def copy_file_timed(src: str | Path, dst: str | Path, chunk_mb: int = 16) -> float:
    """Copy one file (via .part), returns MB/s."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    t0 = time.time()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, chunk_mb << 20)
    tmp.replace(dst)
    return src.stat().st_size / 1e6 / max(time.time() - t0, 1e-6)


def copy_archive_local(zip_path: str | Path, dest_dir: str | Path, margin_gb: float = 15.0,
                       logger=None) -> Path | None:
    """Copy parts + .zip to dest_dir and return the local .zip, or None if the disk is too small.

    Parts already there with the right size are skipped (resumable).
    """
    log = logger.info if logger else print
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = archive_part_files(zip_path)
    todo = [f for f in files if not ((dest_dir / f.name).exists()
                                     and (dest_dir / f.name).stat().st_size == f.stat().st_size)]
    need_gb = sum(f.stat().st_size for f in todo) / 1e9
    free_gb = shutil.disk_usage(dest_dir).free / 1e9
    if need_gb > free_gb - margin_gb:
        log(f"  NOT copying locally: need {need_gb:.1f} GB, free {free_gb:.1f} GB "
            f"(keeping {margin_gb} GB spare). Falling back to parallel extraction from Drive.")
        return None
    log(f"  copying {len(todo)}/{len(files)} archive files ({need_gb:.1f} GB) to {dest_dir}")
    t0 = time.time()
    for i, f in enumerate(todo):
        mbps = copy_file_timed(f, dest_dir / f.name)
        log(f"  [{i + 1}/{len(todo)}] {f.name}: {f.stat().st_size / 1e9:.2f} GB at {mbps:.0f} MB/s")
    if todo:
        secs = max(time.time() - t0, 1e-6)
        log(f"  copy done: {need_gb:.1f} GB in {secs / 60:.1f} min ({need_gb * 1e3 / secs:.0f} MB/s)")
    return dest_dir / Path(zip_path).name
