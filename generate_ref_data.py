#!/usr/bin/env python3
"""generate_ref_data.py — Generate C comparison test data using updated gtcrn_full.py"""
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtcrn_full import *

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_version')

def main():
    np.random.seed(42)
    print("=" * 60)
    print("  GTCRN Python Reference — Test Data Generator (v2)")
    print("=" * 60)

    # Generate 3 test frames
    frames = []
    for f in range(3):
        real = (np.random.randn(257) * 0.005).astype(np.float32)
        imag = (np.random.randn(257) * 0.005).astype(np.float32)
        t = np.arange(257) * (2*np.pi/257)
        real += 0.01 * np.sin(t * (5 + f*3)).astype(np.float32)
        imag += 0.01 * np.cos(t * (5 + f*3) + 1.5).astype(np.float32)
        frames.append((real, imag))

    state = init_state()
    all_crm = []

    for f, (real, imag) in enumerate(frames):
        spec = real + 1j * imag
        crm_float, state = infer_frame(spec, state)
        crm = (crm_float * 1048576.0).astype(np.int32)  # float → s32f20
        all_crm.append(crm)

        max_crm = np.max(np.abs(crm))
        print(f"\n  Frame {f+1}:")
        print(f"    CRM max:     {max_crm:10.1f}  ({max_crm/1048576:.6f} float)")
        print(f"    CRM[0..3] I: {crm[0,0]:8d} {crm[0,1]:8d} {crm[0,2]:8d} {crm[0,3]:8d}")
        print(f"    CRM[0..3] Q: {crm[1,0]:8d} {crm[1,1]:8d} {crm[1,2]:8d} {crm[1,3]:8d}")

    # Save frame 1 input/output
    real0, imag0 = frames[0]
    input_data = np.concatenate([real0, imag0]).astype(np.float32)
    input_path = os.path.join(OUT_DIR, 'ref_frame_input.bin')
    input_data.tofile(input_path)
    print(f"\n  Saved: {input_path} ({input_data.nbytes} bytes)")

    crm0 = all_crm[0]
    crm_data = crm0.astype(np.int32).ravel()
    crm_path = os.path.join(OUT_DIR, 'ref_frame_crm.bin')
    crm_data.tofile(crm_path)
    print(f"  Saved: {crm_path} ({crm_data.nbytes} bytes)")

    # Multi-frame
    all_input = np.concatenate([np.concatenate([r, i]) for r, i in frames]).astype(np.float32)
    multi_in_path = os.path.join(OUT_DIR, 'ref_3frame_input.bin')
    all_input.tofile(multi_in_path)
    print(f"  Saved: {multi_in_path} ({all_input.nbytes} bytes)")

    all_crm_data = np.stack(all_crm).astype(np.int32).ravel()
    multi_crm_path = os.path.join(OUT_DIR, 'ref_3frame_crm.bin')
    all_crm_data.tofile(multi_crm_path)
    print(f"  Saved: {multi_crm_path} ({all_crm_data.nbytes} bytes)")

    print(f"\n{'='*60}")
    print(f"  Done. Files ready for C comparison.")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
