#!/usr/bin/env python3
"""Evidence of (non-)vectorisation: count floating-point multiply/add instructions in the kernel
functions of every binary, split into scalar (…sd) and packed (…pd) forms and by register width.

Reads the disassembly (objdump -d) of mm_blocked (sequential kernel) and row_tile (the per-tile
function the OpenMP version calls). Writes results/vec_evidence.csv and build/disasm_<binary>.txt.
These are static counts (instructions present in the code), not executed-instruction counts.
"""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD, RES = ROOT / "build", ROOT / "results"
BINARIES = ["mm", "mm_scalar", "mm_simd"]
FUNCS = ["mm_blocked", "row_tile"]
PAT = re.compile(r"\b(v?)(mul|add)(sd|pd)\s+([^\n]*)")


def functions(binary):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(BUILD / binary)],
                         capture_output=True, text=True, check=True).stdout
    (BUILD / f"disasm_{binary}.txt").write_text(out)
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur and line.strip():
            funcs[cur].append(line)
    return funcs


def main():
    RES.mkdir(exist_ok=True)
    rows = []
    for b in BINARIES:
        f = functions(b)
        for fn in FUNCS:
            body = "\n".join(f.get(fn, []))
            cnt = dict(scalar=0, packed_xmm=0, packed_ymm=0, packed_zmm=0)
            for _, _, kind, ops in PAT.findall(body):
                if kind == "sd":
                    cnt["scalar"] += 1
                elif "%zmm" in ops:
                    cnt["packed_zmm"] += 1
                elif "%ymm" in ops:
                    cnt["packed_ymm"] += 1
                else:
                    cnt["packed_xmm"] += 1
            rows.append(dict(binary=b, function=fn, present=fn in f, **cnt))
    with open(RES / "vec_evidence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
