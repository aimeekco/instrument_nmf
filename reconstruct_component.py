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

if __name__ == "__main__":
    
    # KL-NMF
    print("--- Running KL-NMF Component Reconstruction ---")
    kl_result_file = "nmf_results/20251115_173733_train_k30_kl_20tracks/nmf_kl_Actions - One Minute Smile_20251115_172412.pkl"
    kl_track_name = "Actions - One Minute Smile"
    
    if os.path.exists(kl_result_file):
        reconstruct_component_audio(
            track_name=kl_track_name, 
            pkl_path=kl_result_file, 
            component_indices=[0, 1, 2] 
        )
    else:
        print(f"KL-NMF result file not found: {kl_result_file}. Skipping KL reconstruction.")

    # IS-NMF
    print("\n--- Running IS-NMF Component Reconstruction ---")

    is_result_file = "nmf_results/20251120_103306_train_k30_is_tracks0-19/nmf_kl_ANiMAL - Clinic A_20251120_103820.pkl"
    is_track_name = "ANiMAL - Clinic A"

    if os.path.exists(is_result_file):
        reconstruct_component_audio(
            track_name=is_track_name, 
            pkl_path=is_result_file, 
            component_indices=[0, 1, 2]
        )
    else:
        print(f"IS-NMF result file not found: {is_result_file}. Skipping IS reconstruction.")

    print("\nReconstruction process complete. Check the 'reconstructed_audio' directory.")
