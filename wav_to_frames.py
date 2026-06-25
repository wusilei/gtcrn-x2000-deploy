#!/usr/bin/env python3
"""wav_to_frames.py — WAV → STFT frames .bin for X2000 C inference

Usage:
  python3 wav_to_frames.py <input.wav> <output_frames.bin>
  python3 wav_to_frames.py  # uses default paths
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import sys, os

N_FFT = 512
WIN_LEN = 512
HOP = 256

def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/media/sf_haidesi/haidesi/gtcrn-x2000-deploy/GTCRN_speech_enhance_FPversion/test_wavs/noisy_fileid_5.wav"
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/tmp/x2000_frames_in.bin"

    # Read WAV
    sr, audio = wavfile.read(wav_path)
    audio = audio / 32768.0  # int16 → float

    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel if stereo

    print(f"Input:  {wav_path}")
    print(f"  SR={sr} Hz, len={len(audio)} samples, {len(audio)/sr:.2f}s")

    # STFT: match MATLAB STFT_func(noisy_audio', N_fft, win_len, win_inc, hann_window)
    # scipy stft: f, t, Zxx = stft(x, fs, window, nperseg, noverlap, nfft)
    # MATLAB: win_len=512, win_inc=256 → noverlap = win_len - win_inc = 256
    f, t, Zxx = stft(audio, fs=sr, window='hann', nperseg=WIN_LEN,
                     noverlap=WIN_LEN - HOP, nfft=N_FFT, boundary='zeros')

    # Zxx shape: (n_freq, n_frames) → n_freq=257 (0..256), complex
    n_freq, n_frames = Zxx.shape
    assert n_freq == 257, f"Expected 257 freq bins, got {n_freq}"

    print(f"  STFT:  {n_frames} frames × {n_freq} bins")
    print(f"  Shape: {Zxx.shape}")

    # Write output: int32 N_frames + N_frames * (float32 real[257] + float32 imag[257])
    with open(out_path, 'wb') as f:
        header = np.array([n_frames], dtype=np.int32)
        header.tofile(f)

        for frame_idx in range(n_frames):
            real = Zxx[:, frame_idx].real.astype(np.float32)
            imag = Zxx[:, frame_idx].imag.astype(np.float32)
            real.tofile(f)
            imag.tofile(f)

    file_size = os.path.getsize(out_path)
    expected = 4 + n_frames * 257 * 2 * 4
    print(f"\nOutput: {out_path}")
    print(f"  Size:  {file_size} bytes (expected {expected})")
    if file_size == expected:
        print("  ✓ OK")
    else:
        print("  ✗ SIZE MISMATCH")

if __name__ == '__main__':
    main()
