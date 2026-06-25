#!/usr/bin/env python3
"""frames_to_wav.py — CRM .bin → ISTFT → enhanced WAV

Usage:
  python3 frames_to_wav.py <crm_output.bin> <noisy_ref.wav> <output.wav>
  python3 frames_to_wav.py  # uses default paths
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft
import sys, os

N_FFT = 512
WIN_LEN = 512
HOP = 256
S32F20_SCALE = 1048576.0  # 2^20

def main():
    crm_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/tmp/x2000_crm_out.bin"
    ref_wav = sys.argv[2] if len(sys.argv) > 2 else \
        "/media/sf_haidesi/haidesi/gtcrn-x2000-deploy/GTCRN_speech_enhance_FPversion/test_wavs/noisy_fileid_5.wav"
    out_path = sys.argv[3] if len(sys.argv) > 3 else \
        "/tmp/enhanced_x2000.wav"

    # Read CRM output from X2000
    with open(crm_path, 'rb') as f:
        N_frames = np.fromfile(f, dtype=np.int32, count=1)[0]
        # Each frame: int32_t I[257] + int32_t Q[257] = 514 int32
        data = np.fromfile(f, dtype=np.int32)
        expected = N_frames * 514
        if len(data) != expected:
            print(f"WARNING: expected {expected} values, got {len(data)}")
            N_frames = len(data) // 514

    print(f"Input:  {crm_path}")
    print(f"  Frames: {N_frames}")
    print(f"  Values: {len(data)}")

    # Reshape: (N_frames, 2, 257) → (2, 257, N_frames)
    crm = data.reshape(N_frames, 2, 257).transpose(1, 2, 0)

    # Convert s32f20 → float
    crm_float = crm.astype(np.float64) / S32F20_SCALE

    # Construct complex spectrum from CRM mask
    # CRM = [I_channel, Q_channel] → spec = I + j*Q
    spec = crm_float[0] + 1j * crm_float[1]  # shape: (257, N_frames)

    print(f"  CRM range: [{crm_float.min():.4f}, {crm_float.max():.4f}]")

    # Read reference WAV for length
    sr, audio_ref = wavfile.read(ref_wav)
    audio_ref = audio_ref / 32768.0
    if audio_ref.ndim > 1:
        audio_ref = audio_ref[:, 0]
    ref_len = len(audio_ref)
    print(f"  Ref len: {ref_len} samples, {ref_len/sr:.2f}s")

    # ISTFT: match MATLAB ISTFT_func(cmp_spec, N_fft, win_len, win_inc, hann_window)
    t_out, audio_out = istft(spec, fs=sr, window='hann', nperseg=WIN_LEN,
                             noverlap=WIN_LEN - HOP, nfft=N_FFT,
                             boundary=False, input_onesided=True)

    # Trim/pad to match reference length
    if len(audio_out) > ref_len:
        audio_out = audio_out[:ref_len]
    elif len(audio_out) < ref_len:
        audio_out = np.pad(audio_out, (0, ref_len - len(audio_out)))

    print(f"  Out len: {len(audio_out)} samples")

    # Clamp to [-1, 1]
    audio_out = np.clip(audio_out, -1.0, 1.0)

    # Save as int16 WAV
    audio_int16 = (audio_out * 32767.0).astype(np.int16)
    wavfile.write(out_path, sr, audio_int16)

    print(f"\nOutput: {out_path}")
    print(f"  SR={sr}, len={len(audio_int16)}, RMS={np.sqrt(np.mean(audio_out**2)):.4f}")
    print("  ✓ Done")

if __name__ == '__main__':
    main()
