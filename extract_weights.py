#!/usr/bin/env python3
"""
extract_weights.py — Extract MATLAB .mat weights → C header
============================================================
Reads all .mat files from GTCRN_speech_enhance_FPversion/para_in_mat/
and generates gtcrn_matlab_weights.h with properly quantized C arrays.

Usage: python extract_weights.py
Output: gtcrn_matlab_weights.h (≈550KB)
"""

import numpy as np
from scipy.io import loadmat
import os, sys
from collections import OrderedDict

# ---- Config ----
MATLAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'GTCRN_speech_enhance_FPversion', 'para_in_mat')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'c_version', 'gtcrn_matlab_weights.h')

# Q-format mapping for each weight type
# Based on MATLAB Fix_point.m calls in the source modules
Q_MAP = {
    # Encoder Conv0
    'encoder_en_convs_0_conv_weight': 's32f18',
    'encoder_en_convs_0_conv_bias': 's32f20',
    'encoder_en_convs_0_bn_weight': 'u16f14',
    'encoder_en_convs_0_bn_bias': 's32f20',
    'encoder_en_convs_0_bn_running_mean': 's32f20',
    'encoder_en_convs_0_bn_running_var': 'u16f14',
    'encoder_en_convs_0_act_weight': 's16f14',

    # Encoder Conv1
    'encoder_en_convs_1_conv_weight': 's16f13',
    'encoder_en_convs_1_conv_bias': 's32f20',
    'encoder_en_convs_1_bn_weight': 'u16f14',
    'encoder_en_convs_1_bn_bias': 's32f20',
    'encoder_en_convs_1_bn_running_mean': 's32f20',
    'encoder_en_convs_1_bn_running_var': 'u16f10',
    'encoder_en_convs_1_act_weight': 's16f14',

    # Decoder DeConv0
    'decoder_de_convs_4_conv_weight': 's32f18',
    'decoder_de_convs_4_conv_bias': 's32f20',
    'decoder_de_convs_4_bn_weight': 'u16f14',
    'decoder_de_convs_4_bn_bias': 's32f20',
    'decoder_de_convs_4_bn_running_mean': 's32f20',
    'decoder_de_convs_4_bn_running_var': 'u16f14',

    # Decoder DeConv1
    'decoder_de_convs_3_conv_weight': 's16f13',
    'decoder_de_convs_3_conv_bias': 's32f20',
    'decoder_de_convs_3_bn_weight': 'u16f14',
    'decoder_de_convs_3_bn_bias': 's32f20',
    'decoder_de_convs_3_bn_running_mean': 's32f20',
    'decoder_de_convs_3_bn_running_var': 'u16f14',
    'decoder_de_convs_3_act_weight': 's16f14',
}

# Q-format defaults for pattern-matched names
def guess_q_format(name: str) -> str:
    if name in Q_MAP:
        return Q_MAP[name]

    # Encoder GT-Conv patterns (indices 2,3,4)
    if 'encoder_en_convs_' in name:
        idx = name.split('_')[3]  # '2', '3', '4'
        if 'depth_conv_weight' in name:
            return 's16f13'
        if 'depth_conv_bias' in name:
            return 's32f20'
        if 'depth_bn_weight' in name:
            return 'u16f14'
        if 'depth_bn_bias' in name:
            return 's32f20'
        if 'depth_bn_running_mean' in name:
            return 's32f20'
        if 'depth_bn_running_var' in name:
            return 'u16f10'
        if 'depth_act_weight' in name:
            return 's16f14'
        if 'point_conv1_weight' in name or 'point_conv2_weight' in name:
            return 's16f13'
        if 'point_conv1_bias' in name or 'point_conv2_bias' in name:
            return 's32f20'
        if 'point_bn1_weight' in name or 'point_bn2_weight' in name:
            return 'u16f14'
        if 'point_bn1_bias' in name or 'point_bn2_bias' in name:
            return 's32f20'
        if 'point_bn1_running_mean' in name or 'point_bn2_running_mean' in name:
            return 's32f20'
        if 'point_bn1_running_var' in name:
            return 'u16f13'
        if 'point_bn2_running_var' in name:
            return 'u16f14'
        if 'point_act_weight' in name:
            return 's16f14'
        if 'tra_att_gru_weight_ih' in name or 'tra_att_gru_weight_hh' in name:
            return 's16f13'
        if 'tra_att_gru_bias_ih' in name or 'tra_att_gru_bias_hh' in name:
            return 's32f20'
        if 'tra_att_fc_weight' in name:
            return 's16f13'
        if 'tra_att_fc_bias' in name:
            return 's32f20'

    # Decoder GT-DeConv patterns (indices 0,1,2)
    if 'decoder_de_convs_' in name:
        parts = name.split('_')
        idx = parts[3]  # '0', '1', '2'
        if 'depth_conv_weight' in name:
            return 's16f12'  # decoder uses s16f12!
        if 'depth_conv_bias' in name:
            return 's32f20'
        if 'depth_bn_weight' in name:
            return 'u16f14'
        if 'depth_bn_bias' in name:
            return 's32f20'
        if 'depth_bn_running_mean' in name:
            return 's32f20'
        if 'depth_bn_running_var' in name:
            return 'u16f12'  # decoder uses u16f12!
        if 'depth_act_weight' in name:
            return 's16f14'
        if 'point_conv1_weight' in name or 'point_conv2_weight' in name:
            return 's16f13'
        if 'point_conv1_bias' in name or 'point_conv2_bias' in name:
            return 's32f20'
        if 'point_bn1_weight' in name or 'point_bn2_weight' in name:
            return 'u16f14'
        if 'point_bn1_bias' in name or 'point_bn2_bias' in name:
            return 's32f20'
        if 'point_bn1_running_mean' in name or 'point_bn2_running_mean' in name:
            return 's32f20'
        if 'point_bn1_running_var' in name or 'point_bn2_running_var' in name:
            return 'u16f14'
        if 'point_act_weight' in name:
            return 's16f14'
        if 'tra_att_gru_weight_ih' in name or 'tra_att_gru_weight_hh' in name:
            return 's16f13'
        if 'tra_att_gru_bias_ih' in name or 'tra_att_gru_bias_hh' in name:
            return 's32f18'  # decoder TRA bias uses s32f18!
        if 'tra_att_fc_weight' in name:
            return 's16f13'
        if 'tra_att_fc_bias' in name:
            return 's32f20'

    # DPRNN patterns
    if 'dpgrnn' in name:
        if '_weight_' in name:
            return 's16f12'  # GRU weights
        if '_bias_' in name:
            return 's16f10'  # GRU bias
        if 'fc_weight' in name:
            return 's16f13'
        if 'fc_bias' in name:
            return 's32f20'
        if 'ln_weight' in name:
            return 's16f12'
        if 'ln_bias' in name:
            return 's32f20'

    # ERB weights
    if 'erb_' in name:
        return 'u16f15'

    # Default
    print(f"WARNING: unknown Q-format for {name}, using s32f20")
    return 's32f20'


def quantize_array(arr: np.ndarray, q_set: str) -> np.ndarray:
    """Quantize array to the given fixed-point format, return integer array."""
    nbits = int(q_set[-2:])
    signed = q_set[0] == 's'
    scale = 2 ** nbits

    if signed:
        return np.round(arr * scale).astype(np.int32 if nbits >= 18 else np.int16)
    else:
        return np.round(np.clip(arr, 0, None) * scale).astype(np.uint32 if nbits >= 18 else np.uint16)


def get_c_type(q_set: str) -> str:
    """Get C type string for a Q-format."""
    nbits = int(q_set[-2:])
    signed = q_set[0] == 's'
    if nbits >= 18:
        return 'int32_t' if signed else 'uint32_t'
    else:
        return 'int16_t' if signed else 'uint16_t'


def array_to_c(arr: np.ndarray, name: str, q_set: str) -> str:
    """Convert numpy array to C array literal string."""
    ctype = get_c_type(q_set)
    flat = arr.ravel()

    lines = []
    lines.append(f'/* {name}: shape {arr.shape}, Q-format: {q_set}, {arr.size} elements */')
    lines.append(f'static const {ctype} {name}[{arr.size}] = {{')

    # Format as hex or decimal
    vals_per_line = 8
    for i in range(0, len(flat), vals_per_line):
        chunk = flat[i:i+vals_per_line]
        line = '    ' + ', '.join(str(int(v)) for v in chunk)
        if i + vals_per_line < len(flat):
            line += ','
        lines.append(line)

    lines.append('};')
    return '\n'.join(lines)


def load_and_prep(name: str) -> np.ndarray:
    """Load a .mat file and prepare the array."""
    filepath = os.path.join(MATLAB_DIR, name)
    data = loadmat(filepath)
    for key in data:
        if not key.startswith('__'):
            arr = data[key]
            # Remove squeeze dims for 1D bias/mean/var
            return arr
    raise KeyError(f"No data found in {filepath}")


def generate_header():
    """Main generator function."""
    if not os.path.isdir(MATLAB_DIR):
        print(f"ERROR: MATLAB para_in_mat directory not found: {MATLAB_DIR}")
        sys.exit(1)

    # Collect all .mat files
    mat_files = sorted(f for f in os.listdir(MATLAB_DIR) if f.endswith('.mat'))

    # Process .' transpose annotations from MATLAB
    # In MATLAB, weight = (importdata('xxx.mat')).' means transpose
    # We need to handle this per-file

    lines = []
    lines.append('/*')
    lines.append(' * gtcrn_matlab_weights.h — Auto-generated weight data')
    lines.append(' * Generated by extract_weights.py')
    lines.append(f' * Total tensors: {len(mat_files)}')
    lines.append(' */')
    lines.append('')
    lines.append('#ifndef GTCRN_MATLAB_WEIGHTS_H')
    lines.append('#define GTCRN_MATLAB_WEIGHTS_H')
    lines.append('')
    lines.append('#include <stdint.h>')
    lines.append('')
    lines.append('#ifdef __cplusplus')
    lines.append('extern "C" {')
    lines.append('#endif')
    lines.append('')

    total_size = 0
    count = 0

    for fname in mat_files:
        var_name = fname.replace('.mat', '')
        arr = load_and_prep(fname)
        q = guess_q_format(var_name)

        # Pre-compute inverse-sqrt for running_var tensors
        # MATLAB: running_var = 1 ./ (sqrt(var + 1e-5))
        if 'running_var' in var_name:
            arr = 1.0 / np.sqrt(np.maximum(arr, 0) + 1e-5)

        # Handle MATLAB .' transpose
        # Weights/Biases/Means that have .' applied in MATLAB source:
        #   weight = (importdata('xxx.mat')).'
        # Pattern covers: GRU weights (_weight_ih_, _weight_hh_),
        #   fc_weight, tra_att_gru weight/bias, bn_weight, erb_ weight,
        #   bn_bias, bn_running_mean
        needs_transpose = any(x in var_name for x in [
            '_weight_', 'fc_weight', 'tra_att_gru', 'tra_att_fc',
            'bn_weight', 'erb_', 'bn_bias', 'bn_running_mean'
        ])
        if needs_transpose and arr.ndim == 2:
            arr = arr.T  # MATLAB .' = non-conjugate transpose

        # Quantize
        arr_q = quantize_array(arr, q)
        total_size += arr_q.nbytes
        count += 1

        # Generate C array
        c_lines = array_to_c(arr_q, var_name, q)
        lines.append(c_lines)
        lines.append('')

    lines.append(f'/* Total tensors: {count}, total size: {total_size} bytes ({total_size/1024:.1f} KB) */')
    lines.append('')
    lines.append('#ifdef __cplusplus')
    lines.append('}')
    lines.append('#endif')
    lines.append('')
    lines.append('#endif /* GTCRN_MATLAB_WEIGHTS_H */')

    with open(OUTPUT, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated {OUTPUT}")
    print(f"  {count} tensors, {total_size/1024:.1f} KB")


if __name__ == '__main__':
    generate_header()
