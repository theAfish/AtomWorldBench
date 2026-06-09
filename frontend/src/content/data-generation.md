# Data Generation

## Generate datasets from CIF files

```bash
# Generate per-action JSON datasets from CIF files
atomworld generate --cif_folder ./cifs --output_dir ./dataset --num_samples 1000

# (Optional) Download structures from Materials Project
python src/scripts/download_random_mp_data.py --api_key YOUR_KEY --out_path ./cifs --num_entries 500
```

You can also apply actions programmatically — see the [Actions](/actions) page for full examples with both simple and verbose APIs.

## Dashboard data

After adding new benchmark results, regenerate the dashboard data file:

```bash
# From the repo root
python src/scripts/generate_gh_pages_data.py
```

This reads every `results/AtomWorld/simple/<model>/<action>/<timestamp>/evaluation_results.json` and writes `docs/data/simple_metrics.json`. Commit and push both files to update the [Results Dashboard](/).