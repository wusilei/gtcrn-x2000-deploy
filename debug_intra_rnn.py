#!/usr/bin/env python3
"""
debug_intra_rnn.py — Intra_RNN step-by-step debug vs MATLAB golden
===================================================================
Dumps every internal intermediate of Intra_RNN module for comparison
with C and MATLAB golden.

Usage:
  python3 debug_intra_rnn.py           # Run with golden ref input
  python3 debug_intra_rnn.py <input.bin>  # Run with custom input

Outputs to: c_version/dump_py/intra_rnn_*.bin
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtcrn_full import *

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_version', 'dump_py')
os.makedirs(OUT_DIR, exist_ok=True)

def save(name, data):
    path = os.path.join(OUT_DIR, name)
    np.asarray(data, dtype=np.int32).ravel().tofile(path)
    return path

def main():
    print("=" * 60)
    print("  Intra_RNN1 Debug — Step-by-Step")
    print("=" * 60)

    # Load input
    if len(sys.argv) > 1:
        x = np.fromfile(sys.argv[1], dtype=np.int32).reshape(16, 33).astype(np.int64)
        print(f"Loaded input from {sys.argv[1]}")
    else:
        # Use golden ref input
        input_data = np.fromfile(
            os.path.join(os.path.dirname(OUT_DIR), 'ref_frame_input.bin'),
            dtype=np.float32)
        r, i = input_data[:257], input_data[257:514]
        state = init_state()
        x_mag = mag_gen(r, i); x_bm = BM_fixed(x_mag, state[6])
        x_sfe = SFE(x_bm)
        y_conv0 = Conv_block(x_sfe, 0); y_conv1 = Conv_block(y_conv0, 1)
        cp_enc = state[0]; hp_enc = state[1]
        y2, cp_enc[0], hp_enc[0] = GT_Conv(y_conv1, cp_enc[0], hp_enc[0], 1, 0)
        y3, cp_enc[1], hp_enc[1] = GT_Conv(y2, cp_enc[1], hp_enc[1], 2, 1)
        x, cp_enc[2], hp_enc[2] = GT_Conv(y3, cp_enc[2], hp_enc[2], 5, 2)

    save("intra_rnn_input.bin", x)
    print(f"\nInput: shape={x.shape}, range=[{x.min()},{x.max()}]")

    # ============================================================
    # Intra_RNN structure (matching MATLAB dpgrnn_intra.m):
    #   x(16,33) → transpose → split x1(8,33) x2(8,33)
    #   x1 → BiGRU(nHidden=4, bidirectional) → (8+8,33)
    #   x2 → BiGRU(nHidden=4, bidirectional) → (8+8,33)
    #   concat → (16,33) → FC(16→16) → LN → residual add → (16,33)
    # ============================================================
    idx = 1
    p = f'dpgrnn{idx}'
    nH = 4

    xT = x.T  # (33, 16)
    x1 = xT[:, :8]   # (33, 8)
    x2 = xT[:, 8:16] # (33, 8)
    save("intra_rnn_x1.bin", x1.T)  # save as (8,33)
    save("intra_rnn_x2.bin", x2.T)
    print(f"x1: shape={x1.shape}, range=[{x1.min()},{x1.max()}]")
    print(f"x2: shape={x2.shape}, range=[{x2.max()},{x2.max()}]")

    # ── GRU weights ──
    r1_ih = W(f'{p}_intra_rnn_rnn1_weight_ih_l0')
    r1_ih_b = W(f'{p}_intra_rnn_rnn1_bias_ih_l0').ravel()
    r1_hh = W(f'{p}_intra_rnn_rnn1_weight_hh_l0')
    r1_hh_b = W(f'{p}_intra_rnn_rnn1_bias_hh_l0').ravel()
    r1_rih = W(f'{p}_intra_rnn_rnn1_weight_ih_l0_reverse')
    r1_rih_b = W(f'{p}_intra_rnn_rnn1_bias_ih_l0_reverse').ravel()
    r1_rhh = W(f'{p}_intra_rnn_rnn1_weight_hh_l0_reverse')
    r1_rhh_b = W(f'{p}_intra_rnn_rnn1_bias_hh_l0_reverse').ravel()

    r2_ih = W(f'{p}_intra_rnn_rnn2_weight_ih_l0')
    r2_ih_b = W(f'{p}_intra_rnn_rnn2_bias_ih_l0').ravel()
    r2_hh = W(f'{p}_intra_rnn_rnn2_weight_hh_l0')
    r2_hh_b = W(f'{p}_intra_rnn_rnn2_bias_hh_l0').ravel()
    r2_rih = W(f'{p}_intra_rnn_rnn2_weight_ih_l0_reverse')
    r2_rih_b = W(f'{p}_intra_rnn_rnn2_bias_ih_l0_reverse').ravel()
    r2_rhh = W(f'{p}_intra_rnn_rnn2_weight_hh_l0_reverse')
    r2_rhh_b = W(f'{p}_intra_rnn_rnn2_bias_hh_l0_reverse').ravel()

    fc_w = W(f'{p}_intra_fc_weight').T
    fc_b = W(f'{p}_intra_fc_bias').ravel()
    ln_w = W(f'{p}_intra_ln_weight')
    ln_b = W(f'{p}_intra_ln_bias')

    print(f"\nGRU weights: ih={r1_ih.shape}, hh={r1_hh.shape}")
    print(f"FC weight: {fc_w.shape}, LN weight: {ln_w.shape}")

    # ── BiGRU x1 (forward + reverse) ──
    h0 = np.zeros((33, nH), np.int64)

    # Forward GRU on x1
    g1_fwd, _ = gru(x1, nH, h0.copy(), r1_ih, r1_ih_b, r1_hh, r1_hh_b)
    save("intra_rnn_gr1_fwd.bin", g1_fwd.T)  # (8,33)
    print(f"\nGRU1 fwd: shape={g1_fwd.shape}, range=[{g1_fwd.min()},{g1_fwd.max()}]")

    # Reverse GRU on x1
    x1_rev = x1[::-1]
    g1_rev_raw, _ = gru(x1_rev, nH, h0.copy(), r1_rih, r1_rih_b, r1_rhh, r1_rhh_b)
    g1_rev = g1_rev_raw[::-1]
    save("intra_rnn_gr1_rev.bin", g1_rev.T)
    print(f"GRU1 rev: shape={g1_rev.shape}, range=[{g1_rev.min()},{g1_rev.max()}]")

    x1_gru = np.concatenate([g1_fwd, g1_rev], 1)  # (33, 8)
    save("intra_rnn_x1_gru.bin", x1_gru.T)
    print(f"x1_gru: shape={x1_gru.shape}")

    # ── BiGRU x2 ──
    g2_fwd, _ = gru(x2, nH, h0.copy(), r2_ih, r2_ih_b, r2_hh, r2_hh_b)
    x2_rev = x2[::-1]
    g2_rev_raw, _ = gru(x2_rev, nH, h0.copy(), r2_rih, r2_rih_b, r2_rhh, r2_rhh_b)
    g2_rev = g2_rev_raw[::-1]
    x2_gru = np.concatenate([g2_fwd, g2_rev], 1)  # (33, 8)
    save("intra_rnn_x2_gru.bin", x2_gru.T)
    print(f"x2_gru: shape={x2_gru.shape}")

    # ── Concat ──
    x_gru = np.concatenate([x1_gru, x2_gru], 1)  # (33, 16)
    save("intra_rnn_x_gru.bin", x_gru.T)
    print(f"x_gru: shape={x_gru.shape}, range=[{x_gru.min()},{x_gru.max()}]")

    # ── FC ──
    fc_w_q, _ = FP(fc_w, 's16f13')
    fc_b_q, _ = FP(fc_b, 's32f20')
    # FC: x_gru(33,16) @ fc_w(16,16) → (33,16)
    x_fc = np.round(x_gru.astype(np.float64) @ fc_w_q.astype(np.float64) / 256) + fc_b_q
    x_fc = x_fc.astype(np.int64)
    save("intra_rnn_fc.bin", x_fc.T)
    print(f"FC: shape={x_fc.shape}, range=[{x_fc.min()},{x_fc.max()}]")

    # ── LN ──
    ln_w_q, _ = FP(ln_w, 's16f12')
    ln_b_q, _ = FP(ln_b, 's32f20')
    y_ln = ln_fixed(x_fc, ln_w_q, ln_b_q, -12)
    save("intra_rnn_ln.bin", y_ln.T)
    print(f"LN: shape={y_ln.shape}, range=[{y_ln.min()},{y_ln.max()}]")

    # ── Residual ──
    y_out = (xT + y_ln).T
    save("intra_rnn_output.bin", y_out)
    print(f"Output: shape={y_out.shape}, range=[{y_out.min()},{y_out.max()}]")
    print(f"Output first 8: {y_out.ravel()[:8]}")

    # ── Compare with full Intra_RNN call ──
    y_full = Intra_RNN(x, idx)
    diff = y_out.astype(np.int64).ravel() - y_full.astype(np.int64).ravel()
    print(f"\nManual vs Intra_RNN(): max_diff={np.max(np.abs(diff))}")

    # ── Compare with golden ──
    G = '/home/a/work/c_version/golden'
    g_rnn = np.fromfile(f'{G}/y_rnn1.bin', dtype=np.int32)
    if g_rnn.size == y_out.size:
        diff = y_out.ravel().astype(np.int64) - g_rnn.astype(np.int64)
        rms = np.sqrt(np.mean(g_rnn.astype(np.float64)**2))
        e = np.sqrt(np.mean(diff.astype(np.float64)**2))
        print(f"Python vs Golden SNR: {20*np.log10(rms/(e+1e-10)):.1f} dB")

    print(f"\nAll intermediates dumped to: {OUT_DIR}/intra_rnn_*.bin")
    print("Ready for C comparison via test_layer_dump or dedicated C debug program.")

if __name__ == '__main__':
    main()
