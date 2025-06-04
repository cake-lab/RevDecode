# Corpus Construction

## Steps Overview

1. Corpus Database Construction
2. Corpus Ground Truth Construction
3. Libary Score Computation

## 1. Corpus Database Construction

### Tool: BSim

Description:
Create a Ghidra project to import and analyze binaries used for the corpus, and create a BSim database to contain them.
This database is used by BSim to match functions in the corpus against unknown functions in the unknown binaries.

Usage:
```bash
mkdir -p /worksapce/corpus/ghidra-project \
         /workspace/corpus/bsim-database \
         /workspace/corpus/signatures &&
analyzeHeadless /workspace/corpus/ghidra-project \
    postgres_object_files \
    -import /workspace/binaries/corpus/* &&
bsim createdatabase file:/workspace/corpus/bsim-database/corpus medium_nosize &&
bsim generatesigs \
    ghidra:/workspace/corpus/ghidra-project/postgres_object_files \
    bsim="file:/workspace/corpus/bsim-database/corpus" \
    /workspace/corpus/signatures &&
bsim commitsigs \
    file:/workspace/corpus/bsim-database/corpus \
    /workspace/corpus/signatures
```

Output:
- A Ghidra project under `/workspace/corpus/ghidra-project` containing the imported corpus binaries.
- A list of `sig_` files under `/workspace/corpus/signatures` directory, each containing XML contents including binary inforation and hashed features of the functions in the corpus.
- A BSim database under `/workspace/corpus/bsim-database/corpus` containing the signatures of the corpus binaries.

## 2. Corpus Ground Truth Construction

### Tool: get_ground_truth_for_corpus.py

Description:
Generate ground truth for each candidate function in the corpus by processing the signatures created in the Corpus Database Construction step.
The ground truth is used to calculate the adjancy scores between candidate functions during the graph construction in RevDecode.

Usage:
```bash
python3 get_ground_truth_for_corpus.py \
    --signature_dir /workspace/corpus/signatures \
    --output_file /workspace/corpus/corpus_ground_truth.json
```

Output:
Ground truth for the corpus will be stored in:
```
/workspace/corpus/corpus_ground_truth.json
```

## 3. Library Score Computation

### Tool: generate_library_scores.py

Description:
Generate library scores for binaries in the corpus based on the signatures created in the Corpus Database Construction step.
The library scores are used as one of the weight factors in the RevDecode graph construction.

Usage:
```bash
python3 generate_library_scores.py \
    --signature_dir /workspace/corpus/signatures \
    --output_file /workspace/corpus/library_scores.json
```

Output:
Library scores will be stored in:
```
/workspace/Corpus/library_scores.json
```
