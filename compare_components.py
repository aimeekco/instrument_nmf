import numpy as np
import matplotlib.pyplot as plt
import musdb
import librosa
import pickle
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity

def get_stem_profiles(track, n_fft=2048, hop_length=512):
    """
    Computes the average frequency profile for each stem in the track.
    Returns a dictionary: {stem_name: frequency_vector}
    """
    stems = ['vocals', 'drums', 'bass', 'other']
    profiles = {}
    
    for stem in stems:
        # Get audio for stem
        audio = track.targets[stem].audio
        audio_mono = np.mean(audio, axis=1)
        
        # STFT
        S = np.abs(librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length))
        
        # Compute average profile (mean over time)
        profile = np.mean(S, axis=1)
        
        # Normalize
        profile = profile / (np.max(profile) + 1e-10)
        
        profiles[stem] = profile
        
    return profiles

def compare_and_plot(track_name, pkl_path, output_dir='analysis_results'):
    """
    Loads NMF results and compares learned components to true stem profiles.
    """
    print(f"Analyzing track: {track_name}")
    print(f"Loading NMF model from: {pkl_path}")
    
    # load NMF model (W)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        W = data['W'] # (n_freq, n_components)
        
    # load ground truth
    mus = musdb.DB(root="dataset/musdb18hq", is_wav=True, subsets=["train", "test"])
    track = None
    for t in mus:
        if t.name == track_name:
            track = t
            break
            
    if track is None:
        print(f"Error: Track '{track_name}' not found.")
        return

    # get stem profiles
    print("Computing true stem frequency profiles...")
    stem_profiles = get_stem_profiles(track)
    
    # cosine similarity
    # W columns vs Stem vectors
    # W shape: (F, K)
    # Stem vector shape: (F,)
    
    stems = list(stem_profiles.keys())
    similarities = np.zeros((len(stems), W.shape[1])) # (4, K)
    
    for i, stem in enumerate(stems):
        stem_vec = stem_profiles[stem].reshape(1, -1) # (1, F)
        # Transpose W to (K, F) for sklearn cosine_similarity
        sim = cosine_similarity(stem_vec, W.T) # (1, K)
        similarities[i, :] = sim
        
    os.makedirs(output_dir, exist_ok=True)
    
    # similarity matrix
    plt.figure(figsize=(12, 5))
    plt.imshow(similarities, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.yticks(range(len(stems)), stems)
    plt.xlabel('NMF Component Index')
    plt.title(f'Similarity between True Stems and Learned Components\n({track_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'similarity_matrix_{track_name.replace(" ", "_")}.png'))
    plt.close()
    
    # best matches plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    freqs = np.linspace(0, track.rate/2, W.shape[0])
    
    for i, stem in enumerate(stems):
        best_comp_idx = np.argmax(similarities[i])
        best_comp_score = similarities[i, best_comp_idx]
        
        true_profile = stem_profiles[stem]
        learned_profile = W[:, best_comp_idx]
        
        true_profile /= np.max(true_profile)
        learned_profile /= np.max(learned_profile)
        
        ax = axes[i]
        ax.plot(freqs, true_profile, label=f'True {stem.title()}', alpha=0.7, color='black', linewidth=1.5)
        ax.plot(freqs, learned_profile, label=f'Comp {best_comp_idx} (Sim: {best_comp_score:.2f})', 
                color='red', linestyle='--', alpha=0.8)
        
        ax.set_title(f'{stem.title()} vs Best Match (Comp {best_comp_idx})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10000]) 
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'best_matches_{track_name.replace(" ", "_")}.png'))
    print(f"Saved plots to {output_dir}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, required=True)
    parser.add_argument('--file', type=str, required=True, help='Path to .pkl file')
    args = parser.parse_args()
    
    compare_and_plot(args.track, args.file)
