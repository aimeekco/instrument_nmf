"""
Apply KL divergence NMF to processed spectrograms.

This script demonstrates:
1. Loading preprocessed spectrograms
2. Applying NMF with Kullback-Leibler (KL) divergence
3. Analyzing the learned basis and activation matrices
4. Reconstructing spectrograms from NMF components
5. Saving and visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import musdb
from preprocess_audio import preprocess_track, get_nmf_input
import os
import pickle
from datetime import datetime


def apply_nmf_kl(V, n_components, max_iter=500, init='random', 
                 random_state=42, verbose=0, tol=1e-4, beta_loss='kullback-leibler'):
    """
    Apply NMF with specified divergence on spectrogram matrix V.
    
    Args:
        V: Non-negative magnitude spectrogram [n_frequencies, n_time_frames]
        n_components: Number of components (K in V ≈ W × H)
        max_iter: Maximum number of iterations
        init: Initialization method ('random', 'nndsvd', 'nndsvda', 'nndsvdar')
        random_state: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        tol: Tolerance for stopping criterion
        beta_loss: Divergence measure ('frobenius', 'kullback-leibler', 'itakura-saito', or numeric)
    
    Returns:
        W: Basis matrix [n_frequencies, n_components]
        H: Activation matrix [n_components, n_time_frames]
        model: Fitted NMF model object
        reconstruction_error: Final reconstruction error
    """
    print(f"\nApplying NMF with {beta_loss} divergence...")
    print(f"  Input shape: {V.shape}")
    print(f"  Components: {n_components}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Initialization: {init}")
    
    # Add small epsilon to avoid zeros for Itakura-Saito and beta_loss <= 0
    # This prevents numerical instabilities and division by zero
    epsilon = 1e-10
    V_safe = V + epsilon
    
    # Check for zero values
    n_zeros = (V == 0).sum()
    if n_zeros > 0 and beta_loss in ['itakura-saito', 0]:
        print(f"  Warning: Input contains {n_zeros} zeros. Adding epsilon={epsilon} for numerical stability.")
    
    print(f"  Min value (after epsilon): {V_safe.min():.2e}")
    
    # Create NMF model with specified divergence
    model = NMF(
        n_components=n_components,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
        beta_loss=beta_loss,
        solver='mu',  # Multiplicative update solver (required for beta_loss != 'frobenius')
        verbose=verbose,
        tol=tol,
        alpha_W=0.0,  # No regularization on W (can be adjusted)
        alpha_H='same',  # Same regularization on H
        l1_ratio=0.0  # L2 regularization (0.0), can use 0.5 for elastic net
    )
    
    # Fit model and transform V to get H
    # Note: sklearn's NMF returns W (transform result) and H (components_)
    # For V ≈ W @ H, we need:
    #   W: [n_frequencies, n_components]
    #   H: [n_components, n_time_frames]
    W = model.fit_transform(V_safe)  # This gives us W: [n_frequencies, n_components]
    H = model.components_  # This gives us H: [n_components, n_time_frames]
    
    # Calculate reconstruction error
    reconstruction_error = model.reconstruction_err_
    
    print(f"\nNMF decomposition completed!")
    print(f"  W shape: {W.shape} (basis matrix)")
    print(f"  H shape: {H.shape} (activation matrix)")
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    print(f"  Iterations: {model.n_iter_}")
    
    return W, H, model, reconstruction_error


def reconstruct_spectrogram(W, H):
    """
    Reconstruct spectrogram from NMF components.
    
    Args:
        W: Basis matrix [n_frequencies, n_components]
        H: Activation matrix [n_components, n_time_frames]
    
    Returns:
        V_reconstructed: Reconstructed spectrogram [n_frequencies, n_time_frames]
    """
    V_reconstructed = W @ H
    return V_reconstructed


def compute_kl_divergence(V, V_reconstructed):
    """
    Compute KL divergence between original and reconstructed spectrograms.
    
    KL(V || V_reconstructed) = sum(V * log(V / V_reconstructed) - V + V_reconstructed)
    
    Args:
        V: Original spectrogram
        V_reconstructed: Reconstructed spectrogram
    
    Returns:
        kl_div: KL divergence value
    """
    # Add small epsilon to avoid division by zero and log(0)
    eps = 1e-10
    V_safe = V + eps
    V_recon_safe = V_reconstructed + eps
    
    kl_div = np.sum(V_safe * np.log(V_safe / V_recon_safe) - V_safe + V_recon_safe)
    
    return kl_div


def plot_nmf_components(W, H, frequencies=None, times=None, 
                        n_components_to_plot=10, save_path=None):
    """
    Visualize NMF basis (W) and activation (H) matrices.
    
    Args:
        W: Basis matrix [n_frequencies, n_components]
        H: Activation matrix [n_components, n_time_frames]
        frequencies: Frequency values for y-axis labels
        times: Time values for x-axis labels
        n_components_to_plot: Number of components to visualize
        save_path: Path to save the figure (if provided)
    """
    K = min(n_components_to_plot, W.shape[1])
    
    fig, axes = plt.subplots(K, 2, figsize=(14, 2.5 * K))
    
    if K == 1:
        axes = axes.reshape(1, -1)
    
    for k in range(K):
        # Plot basis vector (frequency content)
        ax_w = axes[k, 0]
        if frequencies is not None:
            ax_w.plot(W[:, k], frequencies)
            ax_w.set_ylabel('Frequency (Hz)')
        else:
            ax_w.plot(W[:, k])
            ax_w.set_ylabel('Frequency bin')
        ax_w.set_xlabel('Magnitude')
        ax_w.set_title(f'Component {k+1}: Basis W[:, {k}]')
        ax_w.grid(True, alpha=0.3)
        
        # Plot activation vector (temporal evolution)
        ax_h = axes[k, 1]
        if times is not None:
            ax_h.plot(times, H[k, :])
            ax_h.set_xlabel('Time (s)')
        else:
            ax_h.plot(H[k, :])
            ax_h.set_xlabel('Time frame')
        ax_h.set_ylabel('Activation')
        ax_h.set_title(f'Component {k+1}: Activation H[{k}, :]')
        ax_h.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved component visualization to: {save_path}")
    
    return fig


def plot_spectrograms(V_original, V_reconstructed, frequencies=None, 
                      times=None, save_path=None):
    """
    Visualize original and reconstructed spectrograms side by side.
    
    Args:
        V_original: Original spectrogram
        V_reconstructed: Reconstructed spectrogram
        frequencies: Frequency values for y-axis
        times: Time values for x-axis
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original spectrogram
    ax0 = axes[0]
    im0 = ax0.imshow(
        20 * np.log10(V_original + 1e-10),  # Convert to dB
        aspect='auto', 
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    ax0.set_title('Original Spectrogram (dB)')
    ax0.set_xlabel('Time frame' if times is None else 'Time (s)')
    ax0.set_ylabel('Frequency bin' if frequencies is None else 'Frequency (Hz)')
    plt.colorbar(im0, ax=ax0, label='Magnitude (dB)')
    
    # Reconstructed spectrogram
    ax1 = axes[1]
    im1 = ax1.imshow(
        20 * np.log10(V_reconstructed + 1e-10),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    ax1.set_title('Reconstructed Spectrogram (dB)')
    ax1.set_xlabel('Time frame' if times is None else 'Time (s)')
    ax1.set_ylabel('Frequency bin' if frequencies is None else 'Frequency (Hz)')
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    # Difference (error) - use percentile-based clipping for better visualization
    ax2 = axes[2]
    difference = V_original - V_reconstructed
    
    # Use percentiles to avoid outliers dominating the color scale
    vmin_clip = np.percentile(difference, 1)
    vmax_clip = np.percentile(difference, 99)
    max_abs = max(abs(vmin_clip), abs(vmax_clip))
    
    im2 = ax2.imshow(
        difference,
        aspect='auto',
        origin='lower',
        cmap='RdBu',
        interpolation='nearest',
        vmin=-max_abs,
        vmax=max_abs
    )
    
    # Add statistics to title
    mean_abs_error = np.mean(np.abs(difference))
    ax2.set_title(f'Difference (Original - Reconstructed)\nMAE: {mean_abs_error:.4f}')
    ax2.set_xlabel('Time frame' if times is None else 'Time (s)')
    ax2.set_ylabel('Frequency bin' if frequencies is None else 'Frequency (Hz)')
    plt.colorbar(im2, ax=ax2, label='Magnitude difference\n(clipped at 1-99 percentile)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram comparison to: {save_path}")
    
    return fig


def save_nmf_results(W, H, model, metadata, save_dir='nmf_results'):
    """
    Save NMF results to disk.
    
    Args:
        W: Basis matrix
        H: Activation matrix
        model: Fitted NMF model
        metadata: Dictionary with additional information
        save_dir: Directory to save results
    
    Returns:
        save_path: Path where results were saved
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nmf_kl_{metadata.get('track_name', 'unknown')}_{timestamp}.pkl"
    save_path = os.path.join(save_dir, filename)
    
    results = {
        'W': W,
        'H': H,
        'model_params': model.get_params(),
        'reconstruction_error': model.reconstruction_err_,
        'n_iter': model.n_iter_,
        'metadata': metadata
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved NMF results to: {save_path}")
    
    return save_path


def analyze_component_energy(H):
    """
    Analyze the energy distribution across NMF components.
    
    Args:
        H: Activation matrix [n_components, n_time_frames]
    
    Returns:
        energy_per_component: Energy of each component
        sorted_indices: Indices sorted by energy (descending)
    """
    # Energy is the sum of activation values for each component
    energy_per_component = np.sum(H, axis=1)
    sorted_indices = np.argsort(energy_per_component)[::-1]  # Descending order
    
    print("\nComponent Energy Analysis:")
    print("-" * 50)
    for i, idx in enumerate(sorted_indices[:10], 1):
        energy = energy_per_component[idx]
        percentage = 100 * energy / energy_per_component.sum()
        print(f"  Rank {i}: Component {idx} - Energy: {energy:.2f} ({percentage:.2f}%)")
    
    return energy_per_component, sorted_indices


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("KL Divergence NMF on MUSDB18-HQ Dataset")
    print("=" * 70)
    
    # Initialize musdb dataset
    mus = musdb.DB(root="dataset/musdb18hq", is_wav=True, subsets="train")
    print(f"\nLoaded {len(mus)} tracks from training set")
    
    # Select a track to process
    track_idx = 0
    track = mus[track_idx]
    print(f"\nProcessing track {track_idx}: {track.name}")
    print(f"  Artist: {track.artist}")
    print(f"  Title: {track.title}")
    
    # Preprocess audio to get spectrograms
    print("\n" + "-" * 70)
    print("Step 1: Preprocessing audio to magnitude spectrogram")
    print("-" * 70)
    spectrograms = preprocess_track(track, n_fft=2048, hop_length=512, 
                                    power=False, log=False)
    
    # Get the magnitude spectrogram for mixture
    V = get_nmf_input(spectrograms, stem='mixture')
    frequencies = spectrograms['mixture']['frequencies']
    times = spectrograms['mixture']['times']
    
    print(f"\nMagnitude spectrogram matrix V:")
    print(f"  Shape: {V.shape}")
    print(f"  Min: {V.min():.6f}, Max: {V.max():.6f}, Mean: {V.mean():.6f}")
    
    # Apply NMF with KL divergence
    print("\n" + "-" * 70)
    print("Step 2: Applying KL Divergence NMF")
    print("-" * 70)
    
    n_components = 30  # Number of basis functions
    W, H, model, recon_error = apply_nmf_kl(
        V, 
        n_components=n_components,
        max_iter=500,
        init='nndsvda',  # 'nndsvda' often gives better results than 'random'
        random_state=42,
        verbose=1,
        tol=1e-4
    )
    
    # Reconstruct spectrogram
    print("\n" + "-" * 70)
    print("Step 3: Reconstructing spectrogram")
    print("-" * 70)
    V_reconstructed = reconstruct_spectrogram(W, H)
    kl_div = compute_kl_divergence(V, V_reconstructed)
    
    print(f"\nReconstruction quality:")
    print(f"  KL divergence: {kl_div:.6f}")
    print(f"  Mean absolute error: {np.mean(np.abs(V - V_reconstructed)):.6f}")
    print(f"  Relative error: {np.linalg.norm(V - V_reconstructed) / np.linalg.norm(V):.6f}")
    
    # Analyze components
    print("\n" + "-" * 70)
    print("Step 4: Analyzing NMF components")
    print("-" * 70)
    energy, sorted_idx = analyze_component_energy(H)
    
    # Create output directory
    output_dir = "nmf_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize results
    print("\n" + "-" * 70)
    print("Step 5: Visualizing results")
    print("-" * 70)
    
    # Plot components
    components_fig = plot_nmf_components(
        W, H, 
        frequencies=frequencies,
        times=times,
        n_components_to_plot=10,
        save_path=os.path.join(output_dir, f"nmf_components_{track.name.replace('/', '_')}.png")
    )
    
    # Plot spectrograms
    spectrograms_fig = plot_spectrograms(
        V, V_reconstructed,
        frequencies=frequencies,
        times=times,
        save_path=os.path.join(output_dir, f"nmf_spectrograms_{track.name.replace('/', '_')}.png")
    )
    
    # Save results
    print("\n" + "-" * 70)
    print("Step 6: Saving results")
    print("-" * 70)
    
    metadata = {
        'track_name': track.name,
        'track_artist': track.artist,
        'track_title': track.title,
        'n_components': n_components,
        'spectrogram_shape': V.shape,
        'n_fft': 2048,
        'hop_length': 512,
        'sample_rate': spectrograms['mixture']['sample_rate'],
        'kl_divergence': kl_div
    }
    
    save_path = save_nmf_results(W, H, model, metadata, save_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("NMF Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - NMF components: {save_path}")
    print(f"  - Component visualizations: nmf_components_*.png")
    print(f"  - Spectrogram comparison: nmf_spectrograms_*.png")
    
    # Print quality metrics
    print("\n" + "=" * 70)
    print("Quality Assessment:")
    print("=" * 70)
    mean_abs_error = np.mean(np.abs(V - V_reconstructed))
    relative_error = np.linalg.norm(V - V_reconstructed) / np.linalg.norm(V)
    print(f"  Mean Absolute Error: {mean_abs_error:.6f}")
    print(f"  Relative Error (Frobenius): {relative_error:.6f}")
    print(f"  Compression ratio: {V.size / (W.size + H.size):.2f}x")
    print(f"  Original size: {V.size:,} values")
    print(f"  Compressed size: {W.size + H.size:,} values (W + H)")
    
    if relative_error < 0.2:
        print("\n  ✓ Excellent reconstruction quality!")
    elif relative_error < 0.4:
        print("\n  ✓ Good reconstruction quality.")
    else:
        print("\n  ⚠ Moderate reconstruction - consider more components.")
    
    # Show plots
    plt.show()
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("1. Try different numbers of components (n_components)")
    print("2. Experiment with initialization methods ('random', 'nndsvda', 'nndsvdar')")
    print("3. Process multiple tracks and compare results")
    print("4. Add regularization (alpha_W, alpha_H, l1_ratio)")
    print("5. Use log-magnitude spectrograms for better perceptual results")
    print("6. Apply NMF to individual stems (bass, drums, vocals, other)")
    print("7. Implement informed NMF with temporal continuity constraints")
