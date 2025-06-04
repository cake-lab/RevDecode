# Graph Traversal

## Steps Overview
1. Installation
2. Run Graph Traversal

## 1. Installation

Usage:
```bash
make all
```
After building, the following executables will be available in the `revdecode` directory:
- `naive_graph_traversal`
- `fine_grained_graph_traversal`
- `parallel_estimation_graph_traversal`
Each of these executables corresponds to a different graph traversal algorithm implemented in RevDecode.

## 2. Run Graph Traversal

### Tool: fine_grained_graph_traversal

Description:
Applies the fine-grained graph traversal algorithm to the processed rankings.

Usage:
```bash
./fine_grained_graph_traversal /workspace/unknown-binary/processed-rankings 0.7
```
`0.7` is the base adjacency score used in the graph traversal.

Output:
Results will be written to:
```
/workspace/unknown-binary/RevDecode_results_fine_grained_approach
```
Each file in this directory contains the results of the graph traversal for the corresponding unknown binary.
