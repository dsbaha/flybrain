#!/usr/bin/env python3
"""Rebuild BANC .fbc with DNb05_R mirroring AND DNb05 weight scaling.

Background:
- DNb05_R has 0 inputs in BANC v888 (untraced, data gap like DNp09_R).
- DNb05 was saturated at 340Hz, needs scaling to ~140Hz for linear operation.

This script:
1. Mirrors DNb05_L inputs (1,040 edges) to DNb05_R
2. Scales DNb05 input weights by 0.05 (bring 340Hz -> ~140Hz)

Usage:
    python3 scripts/fix_dnb05.py banc888_fixed.fbc banc888_dnb05_fixed.fbc

The weight scaling is a model calibration to bring DNb05 into a
physiological firing range. The mirroring is a data repair for the
untraced right DNb05 (explicit artificial repair, like DNp09_R).
"""
import struct
import numpy as np
import sys

def read_fbc(path):
    with open(path, 'rb') as f:
        magic = f.read(8)
        assert magic == b'FLYBRAIN', f"Bad magic: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        N = struct.unpack('<Q', f.read(8))[0]
        E = struct.unpack('<Q', f.read(8))[0]
        print(f"Version {version}, N={N}, E={E}", flush=True)
        row_ptr = np.fromfile(f, dtype=np.uint32, count=N+1)
        col = np.fromfile(f, dtype=np.uint32, count=E)
        w = np.fromfile(f, dtype=np.int32, count=E)
        root_ids = np.fromfile(f, dtype=np.uint64, count=N)
        inhib = np.fromfile(f, dtype=np.uint8, count=N)
    return N, E, row_ptr, col, w, root_ids, inhib

def write_fbc(path, N, row_ptr, col, w, root_ids, inhib):
    E = len(col)
    with open(path, 'wb') as f:
        f.write(b'FLYBRAIN')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<Q', N))
        f.write(struct.pack('<Q', E))
        row_ptr.astype(np.uint32).tofile(f)
        col.astype(np.uint32).tofile(f)
        w.astype(np.int32).tofile(f)
        root_ids.astype(np.uint64).tofile(f)
        inhib.astype(np.uint8).tofile(f)
    print(f"Wrote {path}: N={N}, E={E}", flush=True)

if __name__ == '__main__':
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, E, row_ptr, col, w, root_ids, inhib = read_fbc(in_path)
    
    DNb05_L_root = 720575941439437586
    DNb05_R_root = 720575941463215827
    idx_L = int(np.where(root_ids == DNb05_L_root)[0][0])
    idx_R = int(np.where(root_ids == DNb05_R_root)[0][0])
    print(f"DNb05_L idx={idx_L}, DNb05_R idx={idx_R}", flush=True)
    
    # Scale factor for DNb05 inputs (bring 340Hz -> ~140Hz)
    SCALE = 0.05
    
    print("Processing DNb05 edges...", flush=True)
    new_edges = []
    for src in range(N):
        start, end = row_ptr[src], row_ptr[src+1]
        for i in range(start, end):
            dst = col[i]
            if dst == idx_L or dst == idx_R:
                w[i] = int(w[i] * SCALE)
                if dst == idx_L:
                    new_edges.append((src, idx_R, w[i]))
        if src % 50000 == 0:
            print(f"  {src}/{N}", flush=True)
    
    print(f"Mirroring {len(new_edges)} edges", flush=True)
    
    src_arr = np.repeat(np.arange(N, dtype=np.uint32), np.diff(row_ptr).astype(np.int64))
    if new_edges:
        new_src = np.array([e[0] for e in new_edges], dtype=np.uint32)
        new_dst = np.array([e[1] for e in new_edges], dtype=np.uint32)
        new_w = np.array([e[2] for e in new_edges], dtype=np.int32)
        src_arr = np.concatenate([src_arr, new_src])
        col = np.concatenate([col, new_dst])
        w = np.concatenate([w, new_w])
    
    order = np.argsort(src_arr, kind='stable')
    src_arr, col, w = src_arr[order], col[order], w[order]
    counts = np.bincount(src_arr, minlength=N).astype(np.uint32)
    row_ptr = np.zeros(N+1, dtype=np.uint32)
    row_ptr[1:] = np.cumsum(counts)
    
    write_fbc(out_path, N, row_ptr, col, w, root_ids, inhib)
    print("Done!", flush=True)
