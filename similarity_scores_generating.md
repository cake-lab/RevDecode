# Similarity Scores Generating

Description:
Following steps describe how to generate simialrity scores for each unknown function in an unknown binary against candidate functions in the corpus,
and return top N candidate functions for each unknown function based on the similarity scores.
We use BSim, a function matching tool provided by Ghidra, to generate these similarity scores.

## Steps Overview

1. Generate Raw Rankings
2. Process Rankings for Graph Construction

## 1. Generate Raw Rankings

### Tool: run_bsim.py, generate_ranking.py

Description:
Builds raw function rankings for the sample binary using BSim.

Usage:
```bash
python3 run_bsim.py --purpose generate_ranking
```

Configuration in `run_bsim.py`:
- Set `ghidra_project` to `/workspace/unknown-binary/ghidra-project`.
- Set `sample_binary_set_binaries_path` to `/workspace/binaries/sample`.

Configuration in `generate_ranking.py`:
- set `DATABASE_URL` to `file:/workspace/corpus/bsim-database/corpus`.
- Set `OUTPUT_FILE_PATH` to `/workspace/unknown-binary/raw-rankings`.

Output:
Raw rankings will be stored in:
```
/workspace/unknown-binary/raw-rankings
```
Each file in this directory contains the raw rankings and corresponding similarity scores and confidence scores for unknown funcitions in each unknown binary.

## 2. Process Rankings for RevDecode

### Tool: process_BSim_results.py

Description:
Add self‑confidence and library scores into raw rankings, then normalizes confidence values for each candidate function in the raw rankings.

Usage:
```bash
python3 process_bsim_results.py \
    --ranking_dir /workspace/unknown-binary/raw-rankings \
    --corpus_ground_truth_file /workspace/corpus/corpus_ground_truth.json \
    --self_confidence_file /workspace/unknown-binary/normalized_self_confidence_scores.json \
    --library_scores_file /workspace/corpus/library_scores.json \
    --output_dir /workspace/unknown-binary/processed-rankings
```

Output:
Processed rankings will be stored in:
```
/workspace/unknown-binary/processed-rankings
```
Each file in this directory contains the processed rankings for unknown functions in the corresponding unknown binary.
Each ranking includes candidate functions, their similarity scores, confidence scores, library scores, and self-confidence score.
