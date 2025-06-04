# RevDecode : Code

This repository contains the code and data for the paper "RevDecode: Enhancing Binary Function Matching with Context-Aware Graph Representations and Relevance Decoding", to be presented at USENIX Security 2025.


### Artifact Overview
This repository includes the following components:
- **binaries**: A simple sample binary and a set of binaries used for corpus building, so you can run RevDecode end-to-end.
- **corpus-construction**: Code for constructing the corpus.
- **revedecode**: Code for graph construction and graph traversal.

# Run RevDecode with Docker
1. Download `Dockerfile` and `docker-compose` into a directory of your choice.
2. Within the same directory, create a directory `src`
3. With Docker running, run the command 
```
docker compose up --build
```
4. Accept the license agreement.
5. Run :
```
docker compose run --rm app bash
```
6. Move all other files in this repository into `/src`. Within the container, `/src` will be `/workspace`
7. Within the container, add Ghidra's support folder to your PATH. 
```
export PATH=/opt/ghidra/support:$PATH
```

### How to Run RevDecode

To run RevDecode, follow these steps:
1. Follow the instructions in the `corpus_construction.md` file to build corpus and get necessary data.
2. Follow the instructions in the `self_confidence_score_generating.md` file to generate self-confidence scores for the unknown binary.
3. Follow the instructions in the `similarity_scores_generating.md` file to generate top N rankings for each unknown function in the unknown binary.
4. Follow the instructions in the `graph_traversal.md` file to run the graph traversal on the GPU and get the results for the graph traversal.


### How to cite our work

Please cite our work as follows:

```
@inproceedings{revdecode2025,
  title     = {RevDecode: Enhancing Binary Function Matching with Context-Aware Graph Representations and Relevance Decoding},
  author    = {Tongwei Ren and Ronghan Che and Guin R. Gilman and Lorenzo De Carli and Robert J. Walls},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  address   = {Seattle, WA},
  publisher = {USENIX Association},
  month     = aug,
  year      = {2025}
}
```

### License
The code in this repository is released under the GNU General Public License version 2. Please see the LICENSE file for more details.


### Bugs and Feedback
If you find any bugs or have feedback, please open an issue in this repository or contact the authors directly.
