# Running NMF on Multiple Songs

### Process first 5 tracks:
```bash
python3 apply_nmf_batch.py --n-tracks 5
```

### Process first 10 tracks without visualizations:
```bash
python3 apply_nmf_batch.py --n-tracks 10 --no-viz
```

### Process tracks 10-20:
```bash
python3 apply_nmf_batch.py --start-idx 10 --n-tracks 10
```

### Process all 100 training tracks (takes a while!):
```bash
python3 apply_nmf_batch.py
```

### Process test set instead of training set:
```bash
python3 apply_nmf_batch.py --subset test --n-tracks 5
```

## All Options

```bash
python3 apply_nmf_batch.py [OPTIONS]

Options:
  --n-tracks N         Number of tracks to process (default: all)
  --start-idx N        Starting track index (default: 0)
  --n-components N     Number of NMF components (default: 30)
  --max-iter N         Maximum NMF iterations (default: 500)
  --no-viz             Skip saving visualizations (faster)
  --output-dir DIR     Output directory (default: nmf_results)
  --subset SUBSET      Dataset subset: train or test (default: train)
```

## Output

For each track, the script saves:
- `nmf_kl_*.pkl` - NMF matrices (W, H) and metadata
- `components_*.png` - Visualization of basis and activation matrices
- `spectrograms_*.png` - Original vs reconstructed spectrograms
- `batch_summary_*.txt` - Overall statistics and quality metrics
