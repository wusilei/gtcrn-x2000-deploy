#!/usr/bin/env python3
"""module_audit.py — Compare individual sub-module outputs: C vs Python
Replicates C's internal layer computation using the SAME primitives as gtcrn_full.py,
feeding the SAME input as C's layer dumps.

Usage: python3 module_audit.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtcrn_full import *

C_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_version', 'dump_pc', 'dump')

def load_c(name):
    return np.fromfile(os.path.join(C_DUMP, name), dtype=np.int32)

def snr(py, c):
    py_f = np.asarray(py, dtype=np.float64).ravel()
    c_f = np.asarray(c, dtype=np.float64).ravel()
    diff = py_f - c_f
    py_rms = np.sqrt(np.mean(py_f**2))
    err_rms = np.sqrt(np.mean(diff**2))
    return 20 * np.log10(py_rms / (err_rms + 1e-10))

def print_result(name, py, c):
    s = snr(py, c)
    md = np.max(np.abs(np.asarray(py, dtype=np.int64).ravel() - np.asarray(c, dtype=np.int64).ravel()))
    bar = '✅' if s > 60 else ('⚠️' if s > 20 else '❌')
    print(f'  {bar} {name:25s} SNR={s:7.1f} dB  max_diff={md}')

print("=" * 70)
print("  Sub-Module Audit: C vs Python (same inputs)")
print("=" * 70)

# ── Test 1: mag_gen ──
real_f32 = np.fromfile(os.path.join(C_DUMP.replace('dump_pc/dump',''), 'ref_frame_input.bin'), dtype=np.float32)[:257]
imag_f32 = np.fromfile(os.path.join(C_DUMP.replace('dump_pc/dump',''), 'ref_frame_input.bin'), dtype=np.float32)[257:514]
py_mag = mag_gen(real_f32, imag_f32)
c_mag = load_c('01_mag.bin')
print_result('01 mag_gen', py_mag, c_mag.reshape(py_mag.shape))

# ── Test 2: BM ──
state = init_state()
erbfc_w = state[6]
py_bm = BM_fixed(py_mag, erbfc_w)  # (3,129)
c_bm = load_c('02_bm.bin')
print_result('02 BM', py_bm, c_bm.reshape(py_bm.shape))

# ── Test 3: SFE ──
py_sfe = SFE(py_bm)  # (9,129)
c_sfe = load_c('03_sfe.bin')
print_result('03 SFE', py_sfe, c_sfe.reshape(py_sfe.shape))

# ── Test 4: Conv0 ──
py_conv0 = Conv_block(py_sfe, 0)  # (16,65)
c_conv0 = load_c('04_conv0.bin')
print_result('04 Conv0', py_conv0, c_conv0.reshape(py_conv0.shape))

# ── Test 5: Conv1 ──
py_conv1 = Conv_block(py_conv0, 1)  # (16,33)
c_conv1 = load_c('05_conv1.bin')
print_result('05 Conv1', py_conv1, c_conv1.reshape(py_conv1.shape))

# ── Test 6-8: GT_Conv ──
# These need history state. For first frame, history is all zeros.
cp_enc = np.zeros((16,16,33), dtype=np.int64)
hp_enc = np.zeros((3,16), dtype=np.int64)

# GT_Conv0 (dil=1)
# The issue with TRA prevents calling GT_Conv directly.
# Instead, let's test sub-components.
print("\n── GT_Conv sub-modules (using C layer inputs as feed) ──")

# Test ddconv2d directly with same inputs as C
# Load C's dd_conv intermediate (if available) or compute from scratch
# We know GT_Conv0 internals: SFE → PC0 → DD-Conv → PC1 → TRA
# Let's test each that we can isolate.

# Test 6: DeConv0 Tanh (end of Decoder)
py_dec0 = DeConv_block(load_c('14_dec1.bin').reshape(16,65),
                        load_c('04_conv0.bin').reshape(16,65), 0)
c_dec0 = load_c('15_dec0.bin')
print_result('15 DeConv0 (Tanh)', py_dec0, c_dec0.reshape(py_dec0.shape))

# Test 7: BS
ierbfc_w = state[7]
py_bs = BS_fixed(py_dec0, ierbfc_w)
c_bs = load_c('16_bs.bin')
print_result('16 BS', py_bs, c_bs.reshape(py_bs.shape))

# Test 8: MASK
py_mask = MASK(py_bs, py_mag[1], py_mag[2])
c_crm = load_c('17_crm.bin')
print_result('17 MASK/CRM', py_mask, c_crm.reshape(py_mask.shape))

# Test 9: DeConv1
py_dec1 = DeConv_block(load_c('13_gtd2.bin').reshape(16,33),
                        load_c('05_conv1.bin').reshape(16,33), 1)
c_dec1 = load_c('14_dec1.bin')
print_result('14 DeConv1', py_dec1, c_dec1.reshape(py_dec1.shape))

print("\n── Direct module comparison done ──")
print("Modules 01-05 and 14-17 are directly comparable.")
print("Modules 06-13 (GT_Conv, RNN, GT_DeConv) have state complexity.")
