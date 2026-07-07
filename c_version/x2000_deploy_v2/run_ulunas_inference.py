"""
Streaming ULUNAS ONNX inference script.
Processes noisy wav files through ulunas_stream_simple.onnx and outputs:
  - *_enhanced.wav : denoised mono audio
  - *_compare.wav  : stereo (left=original noisy, right=enhanced)

Handles both 8kHz (fileid_noise) and 16kHz (war_noise) inputs.
8kHz inputs are resampled to 16kHz for model processing, then back to 8kHz.
"""

import os
import sys
import numpy as np
import onnxruntime
import soundfile as sf
import torch
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "D:/haidesi/haidesi/ul-unas/ulunas_onnx/onnx_models/ulunas_stream_simple.onnx"
TEST_DATA_DIR = "D:/haidesi/haidesi/gtcrn-x2000-deploy/gtcrn_matlab_fixed/test_data"
OUTPUT_BASE_DIR = "D:/haidesi/haidesi/gtcrn-x2000-deploy/gtcrn_matlab_fixed/c_version/x2000_deploy_v2/output_ulunas_field_test"

# Input subfolder -> output subfolder mapping
FOLDER_MAPPING = {
    "fileid_noise": "field_test",
    os.path.join("war_noise", "war_noise"): "war_noise",
}

# STFT parameters (matching ulunas training config)
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
MODEL_SR = 16000

# ONNX model cache dimensions
CONV_CACHE_SIZE = 5358
TFA_CACHE_SIZE = 402
INTER_CACHE_SIZE = 1056


def resample_audio(audio, orig_sr, target_sr):
    """High-quality resampling using scipy's polyphase filter."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    from math import gcd

    try:
        from scipy import signal as scipy_signal

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        print(f"    Resampling {orig_sr}Hz -> {target_sr}Hz (up={up}, down={down})")
        return scipy_signal.resample_poly(audio.astype(np.float64), up, down).astype(np.float32)
    except ImportError:
        # Fallback: simple linear interpolation
        print("    WARNING: scipy not available, using linear interpolation resampling")

        orig_len = len(audio)
        target_len = int(orig_len * target_sr / orig_sr)
        orig_idx = np.linspace(0, orig_len - 1, target_len)
        lo = np.floor(orig_idx).astype(int)
        hi = np.minimum(lo + 1, orig_len - 1)
        frac = orig_idx - lo
        return ((1 - frac) * audio[lo] + frac * audio[hi]).astype(np.float32)


def process_file(session, input_path, output_dir):
    """
    Process a single audio file through the streaming ULUNAS ONNX model.

    Steps:
      1. Read audio
      2. Resample to 16kHz if necessary
      3. STFT (n_fft=512, hop=256, hann window)
      4. Frame-by-frame ONNX inference with state caching
      5. ISTFT to reconstruct enhanced waveform
      6. Resample back to original sample rate if necessary
      7. Save enhanced.wav (mono) and compare.wav (stereo)
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]
    print(f"\n  [{basename}]")

    # ---- 1. Read audio ----
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # use first channel if multi-channel
    original_sr = sr
    original_audio = audio.copy()
    print(f"    Original: {len(audio)} samples @ {sr}Hz = {len(audio)/sr:.3f}s")

    # ---- 2. Resample to 16kHz if needed ----
    if sr != MODEL_SR:
        audio = resample_audio(audio, sr, MODEL_SR)
        sr = MODEL_SR
        print(f"    After resample: {len(audio)} samples @ {sr}Hz")

    # ---- 3. STFT ----
    stft_window = torch.hann_window(WIN_LENGTH)
    x = torch.from_numpy(audio.copy())[None]  # (1, T)
    x_spec_complex = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=stft_window,
        return_complex=True,
        center=True,
    )
    x_spec = torch.view_as_real(x_spec_complex)  # (1, F, T_frames, 2)
    n_frames = x_spec.shape[2]
    print(f"    STFT frames: {n_frames}")

    # ---- 4. Initialize caches ----
    conv_cache = np.zeros((1, CONV_CACHE_SIZE), dtype=np.float32)
    tfa_cache = np.zeros((1, TFA_CACHE_SIZE), dtype=np.float32)
    inter_cache = np.zeros((1, INTER_CACHE_SIZE), dtype=np.float32)

    # ---- 5. Frame-by-frame ONNX inference ----
    inputs_np = x_spec.numpy().astype(np.float32)
    outputs = []

    for i in tqdm(range(n_frames), desc=f"    ONNX inference"):
        out_i, conv_cache, tfa_cache, inter_cache = session.run(
            [],
            {
                "mix": inputs_np[:, :, i : i + 1, :],
                "conv_cache": conv_cache,
                "tfa_cache": tfa_cache,
                "inter_cache": inter_cache,
            },
        )
        outputs.append(out_i)

    # ---- 6. ISTFT ----
    outputs = np.concatenate(outputs, axis=2)  # (1, F, T_frames, 2)
    outputs_torch = torch.from_numpy(outputs)
    outputs_complex = torch.complex(outputs_torch[..., 0], outputs_torch[..., 1])

    enhanced = torch.istft(
        outputs_complex[0],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=stft_window,
        onesided=True,
        center=True,
        length=x.shape[1],
    )
    enhanced = enhanced.detach().cpu().numpy().astype(np.float32)
    print(f"    Enhanced (16kHz): {len(enhanced)} samples")

    # ---- 7. Resample back to original sample rate ----
    if original_sr != MODEL_SR:
        enhanced = resample_audio(enhanced, MODEL_SR, original_sr)
        print(f"    Enhanced ({original_sr}Hz): {len(enhanced)} samples")

    # ---- 8. Align lengths for compare.wav ----
    min_len = min(len(original_audio), len(enhanced))
    original_trimmed = original_audio[:min_len]
    enhanced_trimmed = enhanced[:min_len]

    # ---- 9. Save output files ----
    os.makedirs(output_dir, exist_ok=True)

    enhanced_path = os.path.join(output_dir, f"{basename}_enhanced.wav")
    compare_path = os.path.join(output_dir, f"{basename}_compare.wav")

    # Enhanced: mono
    sf.write(enhanced_path, enhanced_trimmed, original_sr)
    print(f"    Saved: {enhanced_path}")

    # Compare: stereo (left=original noisy, right=enhanced)
    compare_audio = np.stack([original_trimmed, enhanced_trimmed], axis=1)
    sf.write(compare_path, compare_audio, original_sr)
    print(f"    Saved: {compare_path}")


def main():
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading ONNX model: {MODEL_PATH}")
    session = onnxruntime.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    print("Model loaded successfully.")
    print(f"  Inputs:  {[(i.name, i.shape) for i in session.get_inputs()]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")

    # Process each folder
    total_files = 0
    for input_subfolder, output_subfolder in FOLDER_MAPPING.items():
        input_dir = os.path.join(TEST_DATA_DIR, input_subfolder)
        output_dir = os.path.join(OUTPUT_BASE_DIR, output_subfolder)

        if not os.path.isdir(input_dir):
            print(f"\nWARNING: Input directory not found: {input_dir}")
            continue

        wav_files = sorted(
            [
                f
                for f in os.listdir(input_dir)
                if f.lower().endswith(".wav") and not f.endswith("_enhanced.wav") and not f.endswith("_compare.wav")
            ]
        )

        if not wav_files:
            print(f"\nWARNING: No .wav files found in {input_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Files:  {len(wav_files)} wav files")
        print(f"{'='*60}")

        for wav_file in wav_files:
            input_path = os.path.join(input_dir, wav_file)
            try:
                process_file(session, input_path, output_dir)
            except Exception as e:
                print(f"    ERROR processing {wav_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_files += len(wav_files)

    print(f"\n{'='*60}")
    print(f"DONE! Processed {total_files} files.")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
