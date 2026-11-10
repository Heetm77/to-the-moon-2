r"""
Milestone 1 - Step 1: what data do we have?
Lists every symbol, its date range and file count in each data folder,
and prints the columns of one sample file per folder.

Run from PowerShell:
    python D:\Fordham\workspace\research\step1_inventory.py
"""
import os
import re
from collections import defaultdict

import pandas as pd

ROOT = r"D:\Fordham\data\fordham\binance\futures\latest"
DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def read_any(path, nrows=5):
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path).head(nrows)
    if p.endswith((".csv", ".csv.gz", ".gz", ".txt")):
        return pd.read_csv(path, nrows=nrows)
    return None


print("ROOT:", ROOT)
print("Exists:", os.path.isdir(ROOT))
print()

for folder in sorted(os.listdir(ROOT)):
    fpath = os.path.join(ROOT, folder)
    if not os.path.isdir(fpath):
        continue
    print("=" * 70)
    print("FOLDER:", folder)

    files = []
    for dirpath, _, names in os.walk(fpath):
        for n in names:
            files.append(os.path.join(dirpath, n))
    print("Total files:", len(files))

    # Show how files are laid out
    for f in files[:5]:
        print("  e.g.", os.path.relpath(f, fpath))

    # Group by symbol (any path part that looks like XXXUSDT) and collect dates
    by_symbol = defaultdict(list)
    for f in files:
        rel = os.path.relpath(f, fpath)
        sym = next((part for part in re.split(r"[\\/_.\-]", rel)
                    if part.upper().endswith("USDT")), "UNKNOWN")
        m = DATE_RE.search(rel)
        by_symbol[sym.upper()].append(("".join(m.groups()) if m else None))

    print(f"\n{'Symbol':<16}{'Files':>7}  {'First date':<12}{'Last date':<12}")
    for sym in sorted(by_symbol):
        dates = sorted(d for d in by_symbol[sym] if d)
        first = dates[0] if dates else "-"
        last = dates[-1] if dates else "-"
        print(f"{sym:<16}{len(by_symbol[sym]):>7}  {first:<12}{last:<12}")

    # Peek at one file's columns
    for f in files:
        try:
            df = read_any(f)
        except Exception as e:
            print("\nCould not read", f, "->", e)
            continue
        if df is not None:
            print("\nSample file:", os.path.relpath(f, fpath))
            print("Columns:", list(df.columns))
            print(df.head(3).to_string())
            break
    print()
