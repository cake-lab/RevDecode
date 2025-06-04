// parallel_estimation_main.cpp
#include "backtrack.h"
#include "initialize_weight_matrix.h"
#include "parallel_estimation_approach_forward_kernel.h"
#include "pre_forward_kernels.h"
#include "report.h"
#include "utils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

#define NUM_CANDIDATES 512 // Total number of candidates per target function (TOP_K + 1 "Unknown" function)
#define MAX_BLOCKS 432     // The device allowed maximum number of thread blocks that can be utilized at the same time

/**
 * @brief Host function for reading the input, initializing likelihood matrix, performing forward propagation, backtracking, and generating report.
 *
 * @param bonus_factor: Contextual promotion factor applied to transitions meeting certain criterias.
 * @param input_file_path: The local OS directory path for the input .json file.
 * @param output_file_path: The local OS directory path for the output .json report to be saved.
 */
void run_libdecode(double bonus_factor,
                   const string &input_file_path,
                   const string &output_file_path) {
    // Step 1 : Read Input Data

    auto start = chrono::high_resolution_clock::now(); // Log_start
    ifstream input_file(input_file_path);
    if (!input_file.is_open()) {
        cerr << "Error: Unable to open input file: " << input_file_path << endl;
        return;
    }

    nlohmann::json input_json;
    input_file >> input_json;
    input_file.close();

    // Convert the one-chunk JSON content into separate pieces of JSON content,
    // each piece contains the information of the candidates belonged to a single target function.
    vector<nlohmann::json> input_file_contents = input_json.get<vector<nlohmann::json>>();
    auto end = std::chrono::high_resolution_clock::now(); // Log_end
    chrono::duration<double> elapsed = end - start;
    cout << "Read json file elapsed time: " << elapsed.count() << " seconds" << endl;

    // Step 2 : Initialize Data Structures for Device Functions.

    start = chrono::high_resolution_clock::now(); // Log_start

    vector<double> transition_likelihoods;
    vector<CandidateFunction> candidate_function_to_state_machine_map;
    vector<int> num_candidates_per_function;
    int num_targets = 0;
    int num_transitions = 0;

    const double unknown_emission_probability = 0.9;
    const double unknown_uniqueness_score = 0.9;
    const double unknown_promotion_factor = 1.0;

    initialize_machine_for_calculate_probabilities(
        input_file_contents,
        transition_likelihoods,
        candidate_function_to_state_machine_map,
        unknown_emission_probability,
        unknown_uniqueness_score,
        unknown_promotion_factor,
        num_candidates_per_function,
        num_targets,
        num_transitions);

    int total_num_candidates = static_cast<int>(candidate_function_to_state_machine_map.size());

    // Hash candidate functions
    vector<HashedCandidateFunction> hashed_candidates;
    hash_candidates(candidate_function_to_state_machine_map, hashed_candidates);

    HashedCandidateFunction *d_candidates;
    cudaMalloc(&d_candidates, total_num_candidates * sizeof(HashedCandidateFunction));
    cudaMemcpy(d_candidates, hashed_candidates.data(), total_num_candidates * sizeof(HashedCandidateFunction), cudaMemcpyHostToDevice);

    // Allocate transition likelihoods matrix
    double *d_transition_likelihoods;
    cudaMalloc(&d_transition_likelihoods, num_transitions * sizeof(double));
    cudaMemcpy(d_transition_likelihoods, transition_likelihoods.data(), num_transitions * sizeof(double), cudaMemcpyHostToDevice);

    int *d_num_candidates_per_function;
    size_t size = num_candidates_per_function.size() * sizeof(int);
    cudaMalloc(&d_num_candidates_per_function, size);
    cudaMemcpy(d_num_candidates_per_function, num_candidates_per_function.data(), size, cudaMemcpyHostToDevice);

    end = std::chrono::high_resolution_clock::now(); // Log_end
    elapsed = end - start;
    cout << "Initialize likelihood matrix elapsed time: " << elapsed.count() << " seconds" << endl;

    // Step 3 : Launch pre-forward kernels.

    start = chrono::high_resolution_clock::now(); // Log_start

    double *d_state_emission_uniqueness;
    cudaMalloc(&d_state_emission_uniqueness, total_num_candidates * sizeof(double));

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int threads_per_block;
    int num_blocks;

    // Launch the kernel to compute product of `emission_probability` and `uniqueness_score` for each candidate
    threads_per_block = prop.maxThreadsPerBlock;
    num_blocks = (total_num_candidates + threads_per_block - 1) / threads_per_block;

    computeEmissionUniquenessKernel<<<num_blocks, threads_per_block>>>(
        d_candidates, d_state_emission_uniqueness, total_num_candidates);
    cudaDeviceSynchronize();

    // Launch the kernerl to apply contextual bonus factor to each transition
    num_blocks = (num_transitions + threads_per_block - 1) / threads_per_block;
    unsigned int unknown_hash = fnv1a_hash("Unknown");
    preForwardKernel<<<num_blocks, threads_per_block>>>(
        d_transition_likelihoods,
        d_candidates,
        d_state_emission_uniqueness,
        d_num_candidates_per_function,
        bonus_factor,
        unknown_hash,
        num_targets,
        num_transitions);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now(); // Log_end
    elapsed = end - start;
    cout << "Pre-forward kernel elapsed time: " << elapsed.count() << " seconds" << endl;

    // Step 4: Launch forward propagation kernel

    auto start_total = chrono::high_resolution_clock::now(); // Log_start

    // Array storing the highest likelihood for each candidate
    double *d_best_incoming_weight;
    cudaMalloc(&d_best_incoming_weight, total_num_candidates * sizeof(double));
    // cout << "d_best_incoming_weight size when initial: " << total_num_candidates<< endl;

    // Array storing the best predessor index for each candidade
    int *d_best_path;
    cudaMalloc(&d_best_path, total_num_candidates * sizeof(int));

    // Array storing the offset in `transition_likelihoods`
    vector<int> transition_start_indices(num_targets + 1, 0); // +1 for end state

    // Array storing the offset in 'best_incoming_weight'
    vector<int> cumulative_candidates_per_function(num_targets + 1, 0); // +1 for end state

    int cumulative_candidates = 1; // Start from 1 to account for start state
    int cumulative_transitions = 0;

    for (int t = 0; t < num_targets; ++t) { // Iterate through target functions
        cumulative_candidates_per_function[t] = cumulative_candidates;
        cumulative_candidates += num_candidates_per_function[t];

        transition_start_indices[t] = cumulative_transitions;
        if (t == 0) {
            cumulative_transitions += num_candidates_per_function[t];
        } else {
            cumulative_transitions += num_candidates_per_function[t - 1] * num_candidates_per_function[t];
        }
    }

    cumulative_candidates_per_function[num_targets] = cumulative_candidates; // Handle end state
    transition_start_indices[num_targets] = cumulative_transitions;
    cumulative_transitions += num_candidates_per_function[num_targets - 1];

    int *d_transition_start_indices;
    cudaMalloc(&d_transition_start_indices, (num_targets + 1) * sizeof(int));
    cudaMemcpy(d_transition_start_indices, transition_start_indices.data(), (num_targets + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_cumulative_candidates_per_function;
    cudaMalloc(&d_cumulative_candidates_per_function, (num_targets + 1) * sizeof(int));
    cudaMemcpy(d_cumulative_candidates_per_function, cumulative_candidates_per_function.data(), (num_targets + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize arrays for storing rankings
    // Worst-case: Every candidate is in an unique rank group
    int est_ranking_size = (NUM_CANDIDATES * 2 + 1) * total_num_candidates; // Include '-1' and '-2' markers
    // cout << "est_ranking_size: " << est_ranking_size << endl;
    int *d_ranking_values;
    cudaMalloc(&d_ranking_values, est_ranking_size * sizeof(int));

    int *d_ranking_offsets;
    cudaMalloc(&d_ranking_offsets, total_num_candidates * sizeof(int));

    int est_ambiguous_paths_size = (NUM_CANDIDATES + 1) * total_num_candidates; // Include '-2' markers
    int *d_ambiguous_paths;
    cudaMalloc(&d_ambiguous_paths, est_ambiguous_paths_size * sizeof(int));

    int *d_ambiguous_paths_offsets;
    cudaMalloc(&d_ambiguous_paths_offsets, total_num_candidates * sizeof(int));

    int *d_ambiguous_paths_counts;
    cudaMalloc(&d_ambiguous_paths_counts, total_num_candidates * sizeof(int));

    int *d_group_counts;
    cudaMalloc(&d_group_counts, total_num_candidates * sizeof(int));

    vector<int> h_ranking_offsets(total_num_candidates, 0);
    vector<int> h_ambiguous_paths_offsets(total_num_candidates, 0);

    int ranking_offset = 0;
    int ambiguous_offset = 0;
    for (int i = 0; i < total_num_candidates; ++i) {
        h_ranking_offsets[i] = ranking_offset;
        h_ambiguous_paths_offsets[i] = ambiguous_offset;

        ranking_offset += NUM_CANDIDATES * 2 + 1; // Overestimate per candidate
        ambiguous_offset += NUM_CANDIDATES + 1;   // Overestimate per candidate
    }

    cudaMemcpy(d_ranking_offsets, h_ranking_offsets.data(), total_num_candidates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ambiguous_paths_offsets, h_ambiguous_paths_offsets.data(), total_num_candidates * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks_phase0 = MAX_BLOCKS;
    int targets_per_block = (num_targets + num_blocks_phase0 - 1) / num_blocks_phase0; // Make sure no targets are left
    // cout << "targets_per_block: " << targets_per_block << endl;

    // Distribute target functions into thread blocks
    vector<int> block_assignments(num_targets);
    for (int i = 0; i < num_targets; ++i) {
        block_assignments[i] = i / targets_per_block;
    }
    // for (int i = 0; i < num_targets; ++i) {
    //     block_assignments[i] = std::min(i / targets_per_block, MAX_BLOCKS - 1);
    // }
    int *d_block_assignments;
    cudaMalloc(&d_block_assignments, num_targets * sizeof(int));
    cudaMemcpy(d_block_assignments, block_assignments.data(), num_targets * sizeof(int), cudaMemcpyHostToDevice);
    size_t newStackSize = 8192;  
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (err != cudaSuccess) {
        cerr << "Failed to set CUDA device stack size: " << cudaGetErrorString(err) << endl;
    }


    // Launch the forwardKernel for phase 0
    // Merging Logic
    //	Phase 0 (Initial Processing Phase) :
    //		In this phase, each thread block independently processes a set of target functions to initialize the forward
    //		propagation by computing the best incoming weights, paths, and rankings for each candidate.
    //
    // Merging Phases :
    //	Each merging phase reduces the number of active blocks by half by merging boundary targets across adjacent blocks from the previous phase.
    //		If the number of blocks is even, all blocks are merged in pairs.
    //		If the number of blocks is odd, the last block remains unmerged and is forwarded to the next phase.
    //	This process continues until only one block remains.
    //  Each merging phase updates only the boundary target functions by comparing candidates in adjacent pairs. Non-boundary targets
    //  retain their computed values from the previous phase. However, their 'best_incoming_weight' will be computed for forward pass.
    //
    //	Case 1: Even Number of Blocks
    //		When the number of blocks is even, all blocks are merged into pairs of adjacent boundary targets.
    //  Case 2: Odd Number of Blocks
    //		If there is an odd number of blocks, all blocks except the last one are merged in pairs. The last block is deferred to the next phase.
    //		For example, if there are 5 blocks, blocks 0 - 1 and 2 - 3 are merged in pairs, while block 4 is forwarded to the next phase.
    int phase = 0;
    start = chrono::high_resolution_clock::now(); // Log_start
    forwardKernel<<<num_blocks_phase0, NUM_CANDIDATES>>>(
        d_transition_likelihoods,
        d_best_incoming_weight,
        d_best_path,
        d_transition_start_indices,
        d_cumulative_candidates_per_function,
        num_targets,
        total_num_candidates,
        d_ranking_values,
        d_ranking_offsets,
        d_num_candidates_per_function,
        d_ambiguous_paths,
        d_ambiguous_paths_offsets,
        d_ambiguous_paths_counts,
        d_group_counts,
        d_block_assignments,
        phase,
        // Parameters for merging phases
        0,      // 'num_merging_pairs'
        nullptr // 'merging_pair_indices'
    );
    cudaError_t err_2 = cudaGetLastError();
    if (err_2 != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_2) << std::endl;
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now(); // Log_end
    elapsed = end - start;
    cout << "Phase 0 kernel elapsed time: " << elapsed.count() << " seconds" << endl;
    // Debug: Print d_ranking_values content
    // double* test = new double[total_num_candidates];
    // cudaMemcpy(test, d_best_incoming_weight, total_num_candidates * sizeof(double), cudaMemcpyDeviceToHost);

    // std::cout << "d_best_incoming_weight:" << std::endl;
    // for (int i = 0; i < total_num_candidates; ++i) {
    //     std::cout << test[i] << " ";
    //     if ((i + 1) % 16 == 0) std::cout << std::endl; // format for readability
    // }
    // std::cout << std::endl;
    // delete[] test;

    // Launch the forwardKernel for the merging phases
    int remaining_blocks = num_blocks_phase0;
    phase = 1;

    while (remaining_blocks > 1) {
        // Each merging pair is composed of two blocks in previous phase,
        // where the 1st block's right boundary target and the 2nd block's left boundary target are adjacent in the original sequence
        // Each merging pair will be assigned to a new block in current phase
        int num_merging_pairs = remaining_blocks / 2;

        vector<int> h_merging_pair_indices(4 * num_merging_pairs); // Store the target indices for t0, t1, t2, t3

        for (int i = 0; i < num_merging_pairs; ++i) {
            // Map the block id in the previous phase
            int block0 = 2 * i;     // First block in the merging pair
            int block1 = 2 * i + 1; // Second block in the merging pair

            // Find the boundary target functions for block0
            int t0 = -1; // Left boundary of block0 (first target function of block0)
            int t1 = -1; // Right boundary of block0 (last target function of block0)
            for (int j = 0; j < num_targets; ++j) {
                if (block_assignments[j] == block0) {
                    if (t0 == -1) {
                        t0 = j;
                    }
                    t1 = j;
                }
            }

            // Find the boundary target functions for block1
            int t2 = -1; // Left boundary of block1 (first target function of block1)
            int t3 = -1; // Right boundary of block1 (last target function of block1)
            for (int j = 0; j < num_targets; ++j) {
                if (block_assignments[j] == block1) {
                    if (t2 == -1) {
                        t2 = j;
                    }
                    t3 = j;
                }
            }

            h_merging_pair_indices[4 * i] = t0;
            h_merging_pair_indices[4 * i + 1] = t1;
            h_merging_pair_indices[4 * i + 2] = t2;
            h_merging_pair_indices[4 * i + 3] = t3;
        }

        int *d_merging_pair_indices;
        cudaMalloc(&d_merging_pair_indices, 4 * num_merging_pairs * sizeof(int));
        cudaMemcpy(d_merging_pair_indices, h_merging_pair_indices.data(), 4 * num_merging_pairs * sizeof(int), cudaMemcpyHostToDevice);

        start = chrono::high_resolution_clock::now(); // Log_start

        forwardKernel<<<num_merging_pairs, NUM_CANDIDATES>>>(
            d_transition_likelihoods,
            d_best_incoming_weight,
            d_best_path,
            d_transition_start_indices,
            d_cumulative_candidates_per_function,
            num_targets,
            total_num_candidates,
            d_ranking_values,
            d_ranking_offsets,
            d_num_candidates_per_function,
            d_ambiguous_paths,
            d_ambiguous_paths_offsets,
            d_ambiguous_paths_counts,
            d_group_counts,
            d_block_assignments,
            phase,
            num_merging_pairs,
            d_merging_pair_indices);

        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now(); // Log_end
        elapsed = end - start;
        cout << "Merging phase " << phase << " kernel elapsed time: " << elapsed.count() << " seconds" << endl;

        // Update block assignments for next phase
        vector<int> new_block_assignments(num_targets);
        int new_block_id = 0;
        for (int i = 0; i < num_merging_pairs; ++i) { // 'num_merging_pairs' is the number of blocks in current phase
            int block0 = 2 * i;
            int block1 = 2 * i + 1;

            for (int j = 0; j < num_targets; ++j) {
                if (block_assignments[j] == block0 || block_assignments[j] == block1) {
                    new_block_assignments[j] = new_block_id;
                }
            }
            new_block_id++;
        }

        // If the total number of blocks in the current phase is odd, save the last block for next phase to merge
        if (remaining_blocks % 2 == 1) {
            int last_block = remaining_blocks - 1;
            for (int j = 0; j < num_targets; ++j) {
                if (block_assignments[j] == last_block) {
                    new_block_assignments[j] = new_block_id;
                }
            }
            new_block_id++;
        }

        block_assignments = new_block_assignments;

        cudaMemcpy(d_block_assignments, block_assignments.data(), num_targets * sizeof(int), cudaMemcpyHostToDevice);

        remaining_blocks = (remaining_blocks + 1) / 2;
        phase++;

        cudaFree(d_merging_pair_indices);
    }

    auto end_total = std::chrono::high_resolution_clock::now(); // Log_end
    elapsed = end_total - start_total;
    cout << "Forward kernel total elapsed time: " << elapsed.count() << " seconds" << endl;

    // Step 5 : Backtrack

    start = chrono::high_resolution_clock::now(); // Log_start

    // Copy necessary data from device to host
    vector<int> h_ranking_values(est_ranking_size);
    cudaMemcpy(h_ranking_values.data(), d_ranking_values, est_ranking_size * sizeof(int), cudaMemcpyDeviceToHost);

    vector<int> h_ambiguous_paths(est_ambiguous_paths_size);
    cudaMemcpy(h_ambiguous_paths.data(), d_ambiguous_paths, est_ambiguous_paths_size * sizeof(int), cudaMemcpyDeviceToHost);

    vector<int> h_ambiguous_paths_counts(total_num_candidates);
    cudaMemcpy(h_ambiguous_paths_counts.data(), d_ambiguous_paths_counts, total_num_candidates * sizeof(int), cudaMemcpyDeviceToHost);

    vector<vector<vector<int>>> new_ranking = backtrack(
        h_ranking_values,
        h_ranking_offsets,
        h_ambiguous_paths,
        h_ambiguous_paths_offsets,
        h_ambiguous_paths_counts,
        total_num_candidates,
        num_targets);

    end = std::chrono::high_resolution_clock::now(); // Log_end
    elapsed = end - start;
    cout << "Backtrack elapsed time: " << elapsed.count() << " seconds" << endl;

    // Step 6 : Generate output

    start = chrono::high_resolution_clock::now(); // Log_start

    report(
        input_file_contents,
        output_file_path,
        candidate_function_to_state_machine_map,
        new_ranking);
    end = std::chrono::high_resolution_clock::now(); // Log_end
    elapsed = end - start;
    cout << "Generating output json report elapsed time: " << elapsed.count() << " seconds" << endl;

    cudaFree(d_candidates);
    cudaFree(d_transition_likelihoods);
    cudaFree(d_num_candidates_per_function);
    cudaFree(d_state_emission_uniqueness);
    cudaFree(d_best_incoming_weight);
    cudaFree(d_best_path);
    cudaFree(d_transition_start_indices);
    cudaFree(d_cumulative_candidates_per_function);
    cudaFree(d_ranking_values);
    cudaFree(d_ranking_offsets);
    cudaFree(d_ambiguous_paths);
    cudaFree(d_ambiguous_paths_offsets);
    cudaFree(d_ambiguous_paths_counts);
    cudaFree(d_group_counts);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <Path to rankings> <bonus>" << endl;
        return 1;
    }

    string input_dir = argv[1];
    double bonus = stof(argv[2]);

    // Extract the parent directory of the processed_rankings path
    std::filesystem::path input_path(input_dir);
    std::filesystem::path parent_dir = input_path.parent_path();
    std::string parent_str = parent_dir.string();
    // If you need the trailing slash:
    if (parent_str.back() != '/') {
        parent_str += '/';
    }
    std::string output_dir = parent_str + "RevDecode_results_parallel_estimation_approach";
    // Create the output directory if it does not exist
    if (!filesystem::exists(output_dir)) {
        filesystem::create_directories(output_dir);
    }

    // Gather all input files dynamically
    vector<string> input_files;
    for (const auto &entry : filesystem::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            input_files.push_back(entry.path().string());
        }
    }

    if (input_files.empty()) {
        cerr << "No input files found in directory: " << input_dir << endl;
        return 1;
    }

    auto start_time = chrono::high_resolution_clock::now();

    cout << "Parallel estimation approach." << endl;

    for (const auto &input_file_path : input_files) {
        // Extract the base filename without extension
        filesystem::path input_path(input_file_path);
        string filename_stem = input_path.stem().string();

        // Construct the output file path using the original filename
        string output_file_path = output_dir + "/" + filename_stem + "_result_parallel_estimation_approach.json";

        cout << "Processing: " << input_file_path << endl;

        auto file_start_time = chrono::high_resolution_clock::now();

        try {
            run_libdecode(bonus, input_file_path, output_file_path);
            cudaDeviceSynchronize();
            cudaDeviceReset();
        } catch (const exception &e) {
            cerr << "Error processing file " << input_file_path << ": " << e.what() << endl;
            continue;
        }

        auto file_end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> file_elapsed_seconds = file_end_time - file_start_time;
        cout << "Time spent processing " << input_file_path << ": " << file_elapsed_seconds.count() << " seconds" << endl;
        cout << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_seconds = end_time - start_time;
    cout << "Total time spent: " << elapsed_seconds.count() << " seconds" << endl;
    return 0;
}