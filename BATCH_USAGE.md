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

### Using different loss function
```bash
python3 apply_nmf_batch.py --n-tracks 5 --beta-loss itakura-saito
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
  --beta-loss STR      Beta divergence to use: Frobenius, Kullback-Leibler, Itakura-Saito (default: Kullback-Leibler)
```

## Output

Each batch run creates a timestamped directory with all parameters:

```
nmf_results/
└── 20251120_143022_train_k30_kl_tracks0-4/
    ├── batch_config.txt              # Run parameters
    ├── batch_summary.txt             # Results summary
    ├── nmf_kl_*.pkl                  # NMF matrices (W, H) and metadata
    ├── components_*.png              # Basis and activation visualizations
    └── spectrograms_*.png            # Original vs reconstructed spectrograms
```

**Directory name format:** `{timestamp}_{subset}_k{components}_{divergence}_tracks{start}-{end}`
