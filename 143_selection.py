import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
SAMPLE_SIZE  = 143            # rows to sample
RANDOM_STATE = 42             # reproducible shuffle
OUTPUT_DIR   = "sampled_sets" # folder (created if it doesn’t exist)
# ---------------------------------------------------------------------------

script_dir = Path(__file__).resolve().parent      # folder this script lives in
input_dir  = script_dir                           # change if your CSVs are elsewhere
out_dir    = script_dir / OUTPUT_DIR
out_dir.mkdir(exist_ok=True)

for csv_path in input_dir.glob("*_full.csv"):     # look for “…_full.csv”
    df = pd.read_csv(csv_path)

    if len(df) < SAMPLE_SIZE:
        print(f"[WARN] {csv_path.name}: only {len(df)} rows – skipped.")
        continue

    sampled = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    out_file = out_dir / f"{csv_path.stem}_sample{SAMPLE_SIZE}.csv"
    sampled.to_csv(out_file, index=False)

    print(f"[OK]  {csv_path.name} → {out_file.relative_to(script_dir)}")
