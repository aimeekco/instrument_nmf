import numpy as np
import librosa
import soundfile as sf
import pickle
import musdb
import os
from datetime import datetime

def reconstruct_component_audio(track_name, pkl_path, component_indices, output_base_dir="reconstructed_audio", stft_params=None):
    """
    Reconstructs audio for specific NMF components using the original phase.
    
    Args:
        track_name (str): The name of the track (e.g., "ANiMAL - Clinic A").
        pkl_path (str): Full path to the NMF result .pkl file.
        component_indices (list): List of integer indices of components to reconstruct.
        output_base_dir (str): Base directory to save the reconstructed audio.
        stft_params (dict): Dictionary containing 'n_fft', 'hop_length', 'window' if different
                            from default (2048, 512, 'hann').
    """
    print(f"Processing NMF results from: {pkl_path}")
    print(f"  Track: {track_name}")
    print(f"  Components to reconstruct: {component_indices}")
    
    default_stft_params = {'n_fft': 2048, 'hop_length': 512, 'window': 'hann'}
    if stft_params is None:
        stft_params = default_stft_params
    
    n_fft = stft_params['n_fft']
    hop_length = stft_params['hop_length']
    window = stft_params['window']

    # first load the NMF results (W and H)
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            H = data['H']
            model_params = data.get('model_params', {})
            beta_loss = model_params.get('beta_loss', 'unknown')
    except Exception as e:
        print(f"Error loading NMF result file {pkl_path}: {e}")
        return

    batch_name_match = os.path.basename(os.path.dirname(pkl_path))
    output_dir = os.path.join(output_base_dir, batch_name_match)
    os.makedirs(output_dir, exist_ok=True)
    
    # load the original audio to get the Phase
    mus_root = "dataset/musdb18hq"
    mus = musdb.DB(root=mus_root, is_wav=True, subsets=["train", "test"]) # Search both subsets
    
    # find the track in the dataset by its name
    track = None
    for t in mus:
        if t.name == track_name:
            track = t
            break

    if track is None:
        print(f"Error: Track '{track_name}' not found in dataset '{mus_root}'.")
        return
    
    # convert to mono (as NMF was done on mono)
    audio = np.mean(track.audio, axis=1)
    
    # compute STFT (Complex) of the original mixture
    stft_complex = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window)
    
    # extract phase (angle)
    # add epsilon to avoid division by zero errors for phase calculation
    phase = np.exp(1j * np.angle(stft_complex))
    
    print(f"  Original mixture STFT shape: {stft_complex.shape}")
    print(f"  NMF W shape: {W.shape}, H shape: {H.shape}")

    # ensure consistent time frames (can happen due to padding in STFT)
    # if NMF H has more frames than STFT, truncate H
    # if STFT has more frames, truncate phase
    num_frames_V = H.shape[1]
    num_frames_STFT = stft_complex.shape[1]
    
    if num_frames_V > num_frames_STFT:
        print(f"  Warning: H has more frames ({num_frames_V}) than original STFT ({num_frames_STFT}). Truncating H.")
        H = H[:, :num_frames_STFT]
    elif num_frames_STFT > num_frames_V:
        print(f"  Warning: Original STFT has more frames ({num_frames_STFT}) than H ({num_frames_V}). Truncating phase.")
        phase = phase[:, :num_frames_V]

    # reconstruct specific components
    for k in component_indices:
        if k >= W.shape[1]:
            print(f"Skipping index {k}: Model only has {W.shape[1]} components.")
            continue
            
        print(f"  Reconstructing Component {k} ({beta_loss} NMF)...")
        
        # calculate Magnitude for just this component: Outer product of W[:,k] and H[k,:]
        # reshape to ensure 2D arrays: (F, 1) * (1, T)
        V_k = np.outer(W[:, k], H[k, :])
    
        S_k = V_k * phase
        audio_reconstructed = librosa.istft(S_k, hop_length=hop_length, window=window)
        
        safe_track_name = track_name.replace('/', '_').replace(' ', '_')
        safe_pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
        filename = f"{safe_track_name}_component_{k}_{beta_loss}.wav"
        path = os.path.join(output_dir, filename)
        sf.write(path, audio_reconstructed, track.rate)
        print(f"    Saved to: {path}")
    print("-" * 50)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct audio from NMF components.')
    parser.add_argument('--track', type=str, required=True, help='Name of the track (e.g., "Artist - Title")')
    parser.add_argument('--file', type=str, required=True, help='Path to the .pkl result file')
    parser.add_argument('--components', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='List of component indices to reconstruct')
    parser.add_argument('--output-dir', type=str, default='reconstructed_audio', help='Base directory for output')

    args = parser.parse_args()

    if os.path.exists(args.file):
        reconstruct_component_audio(
            track_name=args.track,
            pkl_path=args.file,
            component_indices=args.components,
            output_base_dir=args.output_dir
        )
    else:
        print(f"Error: File not found: {args.file}")
