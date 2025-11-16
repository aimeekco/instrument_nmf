"""
Apply KL divergence NMF to multiple tracks from the MUSDB18-HQ dataset.

This script processes multiple songs in batch mode, saving results for each.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import musdb
from preprocess_audio import preprocess_track, get_nmf_input
import os
import pickle
from datetime import datetime
import argparse
from apply_nmf import (
    apply_nmf_kl, 
    reconstruct_spectrogram, 
    compute_kl_divergence,
    save_nmf_results,
    analyze_component_energy,
    plot_nmf_components,
    plot_spectrograms
)


def process_single_track(track, n_components=30, max_iter=500, 
                         save_visualizations=True, output_dir='nmf_results'):
    """
    Process a single track with NMF.
    
    Args:
        track: musdb Track object
        n_components: Number of NMF components
        max_iter: Maximum iterations for NMF
        save_visualizations: Whether to save plots
        output_dir: Directory to save results
    
    Returns:
        dict: Results summary
    """
    print("\n" + "=" * 70)
    print(f"Processing: {track.name}")
    print(f"  Artist: {track.artist}")
    print(f"  Title: {track.title}")
    print("=" * 70)
    
    try:
        # Preprocess
        spectrograms = preprocess_track(track, n_fft=2048, hop_length=512, 
                                        power=False, log=False)
        V = get_nmf_input(spectrograms, stem='mixture')
        frequencies = spectrograms['mixture']['frequencies']
        times = spectrograms['mixture']['times']
        
        print(f"Spectrogram shape: {V.shape}")
        
        # Apply NMF
        W, H, model, recon_error = apply_nmf_kl(
            V, 
            n_components=n_components,
            max_iter=max_iter,
            init='nndsvda',
            random_state=42,
            verbose=0,  # Silent mode for batch processing
            tol=1e-4
        )
        
        # Reconstruct and evaluate
        V_reconstructed = reconstruct_spectrogram(W, H)
        kl_div = compute_kl_divergence(V, V_reconstructed)
        mae = np.mean(np.abs(V - V_reconstructed))
        rel_error = np.linalg.norm(V - V_reconstructed) / np.linalg.norm(V)
        
        print(f"\nResults:")
        print(f"  Iterations: {model.n_iter_}")
        print(f"  Reconstruction error: {recon_error:.2f}")
        print(f"  KL divergence: {kl_div:.2f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Relative error: {rel_error:.6f}")
        
        # Analyze components
        energy, sorted_idx = analyze_component_energy(H)
        
        # Save results
        metadata = {
            'track_name': track.name,
            'track_artist': track.artist,
            'track_title': track.title,
            'n_components': n_components,
            'spectrogram_shape': V.shape,
            'n_fft': 2048,
            'hop_length': 512,
            'sample_rate': spectrograms['mixture']['sample_rate'],
            'kl_divergence': kl_div,
            'mae': mae,
            'relative_error': rel_error,
            'component_energy': energy.tolist(),
            'sorted_component_indices': sorted_idx.tolist()
        }
        
        save_path = save_nmf_results(W, H, model, metadata, save_dir=output_dir)
        
        # Save visualizations
        if save_visualizations:
            safe_name = track.name.replace('/', '_')
            
            # Plot components (only top 5 for batch processing)
            plot_nmf_components(
                W, H, 
                frequencies=frequencies,
                times=times,
                n_components_to_plot=5,
                save_path=os.path.join(output_dir, f"components_{safe_name}.png")
            )
            plt.close('all')  # Close to free memory
            
            # Plot spectrograms
            plot_spectrograms(
                V, V_reconstructed,
                frequencies=frequencies,
                times=times,
                save_path=os.path.join(output_dir, f"spectrograms_{safe_name}.png")
            )
            plt.close('all')
        
        return {
            'track_name': track.name,
            'success': True,
            'kl_divergence': kl_div,
            'mae': mae,
            'relative_error': rel_error,
            'n_iter': model.n_iter_,
            'save_path': save_path
        }
        
    except Exception as e:
        print(f"\n❌ Error processing {track.name}: {str(e)}")
        return {
            'track_name': track.name,
            'success': False,
            'error': str(e)
        }


def process_batch(n_tracks=None, start_idx=0, n_components=30, max_iter=500, 
                  save_visualizations=True, output_dir='nmf_results',
                  subset='train'):
    """
    Process multiple tracks in batch.
    
    Args:
        n_tracks: Number of tracks to process (None = all)
        start_idx: Starting track index
        n_components: Number of NMF components
        max_iter: Maximum iterations
        save_visualizations: Whether to save plots
        output_dir: Output directory
        subset: 'train' or 'test'
    
    Returns:
        list: Results for all tracks
    """
    print("=" * 70)
    print(f"BATCH NMF PROCESSING - MUSDB18-HQ Dataset")
    print("=" * 70)
    
    # Initialize dataset
    mus = musdb.DB(root="dataset/musdb18hq", is_wav=True, subsets=subset)
    print(f"\nDataset: {subset}")
    print(f"Total tracks available: {len(mus)}")
    
    # Determine tracks to process
    if n_tracks is None:
        end_idx = len(mus)
    else:
        end_idx = min(start_idx + n_tracks, len(mus))
    
    tracks_to_process = list(range(start_idx, end_idx))
    print(f"Processing tracks {start_idx} to {end_idx-1} ({len(tracks_to_process)} tracks)")
    print(f"Components: {n_components}, Max iterations: {max_iter}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process tracks
    results = []
    start_time = datetime.now()
    
    for i, track_idx in enumerate(tracks_to_process, 1):
        track = mus[track_idx]
        print(f"\n[{i}/{len(tracks_to_process)}] Track {track_idx}", end=" ")
        
        result = process_single_track(
            track, 
            n_components=n_components,
            max_iter=max_iter,
            save_visualizations=save_visualizations,
            output_dir=output_dir
        )
        results.append(result)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal tracks: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Time elapsed: {duration:.1f}s ({duration/len(results):.1f}s per track)")
    
    if successful:
        print("\n" + "-" * 70)
        print("Quality Metrics Summary:")
        print("-" * 70)
        avg_kl = np.mean([r['kl_divergence'] for r in successful])
        avg_mae = np.mean([r['mae'] for r in successful])
        avg_rel = np.mean([r['relative_error'] for r in successful])
        avg_iter = np.mean([r['n_iter'] for r in successful])
        
        print(f"Average KL divergence: {avg_kl:.2f}")
        print(f"Average MAE: {avg_mae:.6f}")
        print(f"Average relative error: {avg_rel:.6f}")
        print(f"Average iterations: {avg_iter:.1f}")
        
        print("\nTop 5 best reconstructions (lowest relative error):")
        sorted_results = sorted(successful, key=lambda x: x['relative_error'])
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['track_name']}: {r['relative_error']:.6f}")
    
    if failed:
        print("\n" + "-" * 70)
        print("Failed tracks:")
        print("-" * 70)
        for r in failed:
            print(f"  ❌ {r['track_name']}: {r['error']}")
    
    # Save summary
    summary_path = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Batch NMF Processing Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Subset: {subset}\n")
        f.write(f"Tracks processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Components: {n_components}\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Duration: {duration:.1f}s\n\n")
        
        if successful:
            f.write("-" * 70 + "\n")
            f.write("Results:\n")
            f.write("-" * 70 + "\n")
            for r in successful:
                f.write(f"{r['track_name']}\n")
                f.write(f"  KL divergence: {r['kl_divergence']:.2f}\n")
                f.write(f"  MAE: {r['mae']:.6f}\n")
                f.write(f"  Relative error: {r['relative_error']:.6f}\n")
                f.write(f"  Iterations: {r['n_iter']}\n\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"All results saved to: {output_dir}/")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply KL divergence NMF to multiple tracks from MUSDB18-HQ'
    )
    parser.add_argument(
        '--n-tracks', 
        type=int, 
        default=None,
        help='Number of tracks to process (default: all)'
    )
    parser.add_argument(
        '--start-idx', 
        type=int, 
        default=0,
        help='Starting track index (default: 0)'
    )
    parser.add_argument(
        '--n-components', 
        type=int, 
        default=30,
        help='Number of NMF components (default: 30)'
    )
    parser.add_argument(
        '--max-iter', 
        type=int, 
        default=500,
        help='Maximum NMF iterations (default: 500)'
    )
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Skip saving visualizations (faster)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='nmf_results',
        help='Output directory (default: nmf_results)'
    )
    parser.add_argument(
        '--subset', 
        type=str, 
        default='train',
        choices=['train', 'test'],
        help='Dataset subset (default: train)'
    )
    
    args = parser.parse_args()
    
    # Run batch processing
    results = process_batch(
        n_tracks=args.n_tracks,
        start_idx=args.start_idx,
        n_components=args.n_components,
        max_iter=args.max_iter,
        save_visualizations=not args.no_viz,
        output_dir=args.output_dir,
        subset=args.subset
    )
