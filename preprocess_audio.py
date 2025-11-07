"""
Audio preprocessing for KL divergence NMF on MUSDB18-HQ dataset.

This script demonstrates how to:
1. Load audio files using the musdb package
2. Convert to magnitude spectrogram (non-negative matrix)
3. Prepare data for NMF decomposition
"""

import numpy as np
import librosa
import musdb


def stereo_to_mono(audio):
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        audio: Audio signal, shape (nb_samples, 2) for stereo or (nb_samples,) for mono
    
    Returns:
        y: Mono audio signal (1D array)
    """
    if len(audio.shape) == 2:
        # Stereo: average left and right channels
        y = np.mean(audio, axis=1)
    else:
        # Already mono
        y = audio
    return y


def compute_magnitude_spectrogram(y, sr, n_fft=2048, hop_length=512, 
                                   window='hann', power=False, log=False):
    """
    Compute magnitude spectrogram from audio signal.
    
    Args:
        y: Audio signal (1D array, mono)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        window: Window function type
        power: If True, return power spectrogram (magnitude^2)
        log: If True, return log-magnitude spectrogram
    
    Returns:
        V: Magnitude spectrogram matrix [n_frequencies, n_time_frames]
        frequencies: Frequency values for each bin
        times: Time values for each frame
    """
    # Compute STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    
    # Extract magnitude (non-negative)
    magnitude = np.abs(S)
    
    # Get frequency and time axes
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), 
                                    sr=sr, hop_length=hop_length)
    
    # Apply optional transformations
    if power:
        V = magnitude ** 2
    elif log:
        V = np.log(magnitude + 1e-10)  # Add small epsilon to avoid log(0)
    else:
        V = magnitude
    
    return V, frequencies, times


def preprocess_track(track, n_fft=2048, hop_length=512, 
                     power=False, log=False, mono=True):
    """
    Preprocess a musdb track (mixture + stems) to spectrograms.
    
    Args:
        track: musdb Track object
        n_fft: FFT window size
        hop_length: Hop length for STFT
        power: Use power spectrogram
        log: Use log-magnitude spectrogram
        mono: Convert stereo to mono if True
    
    Returns:
        dict: Dictionary with spectrograms for mixture and each stem
    """
    sr = track.rate
    spectrograms = {}
    
    # Process mixture
    audio = track.audio  # Shape: (nb_samples, 2) for stereo
    if mono:
        y = stereo_to_mono(audio)
    else:
        # For stereo, process each channel separately or use left channel
        y = audio[:, 0] if len(audio.shape) == 2 else audio
    
    V, freqs, times = compute_magnitude_spectrogram(
        y, sr, n_fft=n_fft, hop_length=hop_length, 
        power=power, log=log
    )
    
    spectrograms['mixture'] = {
        'spectrogram': V,  # This is the workable output for NMF
        'frequencies': freqs,
        'times': times,
        'sample_rate': sr,
        'audio_length': len(y) / sr
    }
    
    print(f"Processed mixture: shape={V.shape}, "
          f"duration={len(y)/sr:.2f}s, "
          f"freq_range=[{freqs[0]:.1f}, {freqs[-1]:.1f}] Hz")
    
    # Process stems/targets
    target_names = ['bass', 'drums', 'vocals', 'other']
    for target_name in target_names:
        if target_name in track.targets:
            target_audio = track.targets[target_name].audio
            if mono:
                y_target = stereo_to_mono(target_audio)
            else:
                y_target = target_audio[:, 0] if len(target_audio.shape) == 2 else target_audio
            
            V_target, _, _ = compute_magnitude_spectrogram(
                y_target, sr, n_fft=n_fft, hop_length=hop_length, 
                power=power, log=log
            )
            
            spectrograms[target_name] = {
                'spectrogram': V_target,
                'frequencies': freqs,
                'times': times,
                'sample_rate': sr,
                'audio_length': len(y_target) / sr
            }
            
            print(f"Processed {target_name}: shape={V_target.shape}, "
                  f"duration={len(y_target)/sr:.2f}s")
    
    return spectrograms


def get_nmf_input(spectrograms, stem='mixture'):
    """
    Extract the workable NMF input matrix from preprocessed spectrograms.
    
    Args:
        spectrograms: Dictionary from preprocess_track()
        stem: Which stem to extract ('mixture', 'bass', 'drums', 'vocals', 'other')
    
    Returns:
        V: Non-negative matrix [n_frequencies, n_time_frames] ready for NMF
    """
    if stem not in spectrograms:
        raise ValueError(f"Stem '{stem}' not found in spectrograms. "
                         f"Available: {list(spectrograms.keys())}")
    
    V = spectrograms[stem]['spectrogram']
    
    # Ensure non-negative (should already be, but double-check)
    V = np.maximum(V, 0)
    
    return V


# Example usage
if __name__ == "__main__":
    # Initialize musdb dataset
    # For MUSDB18-HQ (WAV version), use is_wav=True
    # The root path should point to the dataset directory
    mus = musdb.DB(root="dataset/musdb18hq", is_wav=True, subsets="train")
    
    print("=" * 60)
    print("Preprocessing audio for KL divergence NMF")
    print(f"Using musdb package with {len(mus)} tracks")
    print("=" * 60)
    
    # Process first track as example
    track = mus[0]
    print(f"\nProcessing track: {track.name}")
    print(f"Artist: {track.artist}, Title: {track.title}")
    
    # Preprocess with default settings (magnitude spectrogram)
    spectrograms = preprocess_track(track, n_fft=2048, hop_length=512)
    
    # Extract the workable NMF input (magnitude spectrogram of mixture)
    V_mixture = get_nmf_input(spectrograms, stem='mixture')
    
    print("\n" + "=" * 60)
    print("NMF Input Matrix (V)")
    print("=" * 60)
    print(f"Shape: {V_mixture.shape}")
    print(f"  - Rows (frequencies): {V_mixture.shape[0]}")
    print(f"  - Columns (time frames): {V_mixture.shape[1]}")
    print(f"Min value: {V_mixture.min():.6f}")
    print(f"Max value: {V_mixture.max():.6f}")
    print(f"Mean value: {V_mixture.mean():.6f}")
    print(f"Non-negative: {(V_mixture >= 0).all()}")
    
    print("\n" + "=" * 60)
    print("This matrix V is ready for NMF decomposition:")
    print("  V ≈ W × H")
    print("  where W = basis matrix [F, K], H = activation matrix [K, T]")
    print("=" * 60)
    
    # Example: Preprocess with log-magnitude (often better for NMF)
    print("\n" + "=" * 60)
    print("Alternative: Log-magnitude spectrogram")
    print("=" * 60)
    spectrograms_log = preprocess_track(track, n_fft=2048, 
                                        hop_length=512, log=True)
    V_log = get_nmf_input(spectrograms_log, stem='mixture')
    print(f"Log-magnitude shape: {V_log.shape}")
    print(f"Min value: {V_log.min():.6f}")
    print(f"Max value: {V_log.max():.6f}")
    
    # Example: Iterate over multiple tracks
    print("\n" + "=" * 60)
    print("Example: Processing multiple tracks")
    print("=" * 60)
    print("You can iterate over tracks like this:")
    print("  for track in mus:")
    print("      spectrograms = preprocess_track(track)")
    print("      V = get_nmf_input(spectrograms, stem='mixture')")
    print("      # ... do NMF on V ...")
