#!/usr/bin/env python3
"""layer_audit.py — Systematic C vs Python layer-by-layer audit
Dumps all 17 intermediate layers from Python, then compares against C dumps.
"""
import numpy as np
from scipy.io import loadmat
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtcrn_full import *

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_version', 'dump_py')
C_DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_version', 'dump_pc', 'dump')
os.makedirs(OUT_DIR, exist_ok=True)

def save_bin(name, data):
    path = os.path.join(OUT_DIR, name)
    data = np.asarray(data, dtype=np.int32).ravel()
    data.tofile(path)
    return path

def compare_layer(name, py_data, c_file, shape_hint=None):
    """Compare Python vs C layer, return SNR"""
    c_path = os.path.join(C_DUMP_DIR, c_file)
    if not os.path.exists(c_path):
        return None, f"MISSING C dump: {c_file}"
    c_data = np.fromfile(c_path, dtype=np.int32)
    py_flat = np.asarray(py_data, dtype=np.int64).ravel()
    if len(py_flat) != len(c_data):
        return None, f"SIZE MISMATCH: py={len(py_flat)} c={len(c_data)}"

    diff = py_flat.astype(np.int64) - c_data.astype(np.int64)
    max_diff = np.max(np.abs(diff))
    py_rms = np.sqrt(np.mean(py_flat.astype(np.float64)**2))
    err_rms = np.sqrt(np.mean(diff.astype(np.float64)**2))
    snr = 20 * np.log10(py_rms / (err_rms + 1e-10))
    return snr, max_diff

def main():
    # Load input from ref_frame_input.bin
    input_path = os.path.join(os.path.dirname(OUT_DIR), 'ref_frame_input.bin')
    input_data = np.fromfile(input_path, dtype=np.float32)
    real_in = input_data[:257]
    imag_in = input_data[257:514]

    print("=" * 70)
    print("  Layer-by-Layer C vs Python Audit")
    print("=" * 70)
    state = init_state()

    # ── Layer 01: mag_gen ──
    x_mag = mag_gen(real_in, imag_in)
    save_bin("01_mag.bin", x_mag)
    snr, info = compare_layer("01_mag", x_mag, "01_mag.bin")
    print(f" 01_mag:        SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 02: BM ──
    erbfc_w = state[6]
    x_bm = BM_fixed(x_mag, erbfc_w)
    save_bin("02_bm.bin", x_bm)
    snr, info = compare_layer("02_bm", x_bm, "02_bm.bin")
    print(f" 02_bm:         SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 03: SFE ──
    x_sfe = SFE(x_bm)
    save_bin("03_sfe.bin", x_sfe)
    snr, info = compare_layer("03_sfe", x_sfe, "03_sfe.bin")
    print(f" 03_sfe:        SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 04: Conv0 ──
    y_conv0 = Conv_block(x_sfe, 0)
    save_bin("04_conv0.bin", y_conv0)
    snr, info = compare_layer("04_conv0", y_conv0, "04_conv0.bin")
    print(f" 04_conv0:      SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 05: Conv1 ──
    y_conv1 = Conv_block(y_conv0, 1)
    save_bin("05_conv1.bin", y_conv1)
    snr, info = compare_layer("05_conv1", y_conv1, "05_conv1.bin")
    print(f" 05_conv1:      SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 06-08: GT_Conv (encoder) ──
    cp_enc = state[0]; hp_enc = state[1]
    y_conv2, cp_e1, hp_e1 = GT_Conv(y_conv1, cp_enc, hp_enc, 1, 0)
    save_bin("06_gtconv0.bin", y_conv2)
    snr, info = compare_layer("06_gtconv0", y_conv2, "06_gtconv0.bin")
    print(f" 06_gtconv0:    SNR={snr:7.1f} dB  {info if snr is None else ''}")

    y_conv3, cp_e2, hp_e2 = GT_Conv(y_conv2, cp_enc, hp_enc, 2, 1)
    save_bin("07_gtconv1.bin", y_conv3)
    snr, info = compare_layer("07_gtconv1", y_conv3, "07_gtconv1.bin")
    print(f" 07_gtconv1:    SNR={snr:7.1f} dB  {info if snr is None else ''}")

    y_conv4, cp_e3, hp_e3 = GT_Conv(y_conv3, cp_enc, hp_enc, 5, 2)
    save_bin("08_gtconv2.bin", y_conv4)
    snr, info = compare_layer("08_gtconv2", y_conv4, "08_gtconv2.bin")
    print(f" 08_gtconv2:    SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 09-10: GDPRNN ──
    inter1 = state[2]; inter2 = state[3]
    y_rnn1 = Intra_RNN(y_conv4, 1)
    save_bin("09_rnn1.bin", y_rnn1)
    snr, info = compare_layer("09_rnn1", y_rnn1, "09_rnn1.bin")
    print(f" 09_rnn1:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    y_rnn2, inter2_new = Inter_RNN(y_rnn1, inter1, 1)
    save_bin("10a_inter1.bin", y_rnn2)
    snr, info = compare_layer("10a_inter1", y_rnn2, "10_rnn2.bin")
    print(f" 10_rnn2_inter1:SNR={snr:7.1f} dB  {info if snr is None else ''}")

    y_rnn3 = Intra_RNN(y_rnn2, 2)
    y_rnn4, inter2_new2 = Inter_RNN(y_rnn3, inter2, 2)
    save_bin("10_rnn2.bin", y_rnn4)
    snr, info = compare_layer("10_rnn2", y_rnn4, "10_rnn2.bin")
    print(f" 10_rnn2:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 11-13: Decoder GT_DeConv ──
    cp_dec = state[4]; hp_dec = state[5]
    y_d0, cp_d1, hp_d1 = GT_DeConv(y_rnn4, y_conv4, cp_dec, hp_dec, 5, 0)
    save_bin("11_gtd0.bin", y_d0)
    snr, info = compare_layer("11_gtd0", y_d0, "11_gtd0.bin")
    print(f" 11_gtd0:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    y_d1, cp_d2, hp_d2 = GT_DeConv(y_d0, y_conv3, cp_dec, hp_dec, 2, 1)
    save_bin("12_gtd1.bin", y_d1)
    snr, info = compare_layer("12_gtd1", y_d1, "12_gtd1.bin")
    print(f" 12_gtd1:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    y_d2, cp_d3, hp_d3 = GT_DeConv(y_d1, y_conv2, cp_dec, hp_dec, 1, 2)
    save_bin("13_gtd2.bin", y_d2)
    snr, info = compare_layer("13_gtd2", y_d2, "13_gtd2.bin")
    print(f" 13_gtd2:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 14: DeConv1 ──
    y_d3 = DeConv_block(y_d2, y_conv1, 1)
    save_bin("14_dec1.bin", y_d3)
    snr, info = compare_layer("14_dec1", y_d3, "14_dec1.bin")
    print(f" 14_dec1:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 15: DeConv0 ──
    y_dec = DeConv_block(y_d3, y_conv0, 0)
    save_bin("15_dec0.bin", y_dec)
    snr, info = compare_layer("15_dec0", y_dec, "15_dec0.bin")
    print(f" 15_dec0:       SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 16: BS ──
    ierbfc_w = state[7]
    y_bs = BS_fixed(y_dec, ierbfc_w)
    save_bin("16_bs.bin", y_bs)
    snr, info = compare_layer("16_bs", y_bs, "16_bs.bin")
    print(f" 16_bs:         SNR={snr:7.1f} dB  {info if snr is None else ''}")

    # ── Layer 17: MASK/CRM ──
    y_mask = MASK(y_bs, x_mag[1], x_mag[2])
    save_bin("17_crm.bin", y_mask)
    snr, info = compare_layer("17_crm", y_mask, "17_crm.bin")
    print(f" 17_crm:        SNR={snr:7.1f} dB  {info if snr is None else ''}")

    print("=" * 70)
    print(f"Python dumps saved to: {OUT_DIR}")
    print(f"C dumps compared from: {C_DUMP_DIR}")

if __name__ == '__main__':
    main()
