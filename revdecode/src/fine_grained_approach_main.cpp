// fine_grained_approach_main.cpp
#include "backtrack.h"
#include "fine_grained_approach_forward_kernel.h"
#include "initialize_weight_matrix.h"
#include "pre_forward_kernels.h"
#include "report.h"
#include "utils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cfloat>

using namespace std;
using json = nlohmann::json;

#define NUM_CANDIDATES 512 // Total number of candidates per target function (TOP_K + 1 "Unknown" function)

#include <iostream>
#include <vector>

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
    // Step 1: Read Input Data

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

    // Step 2: Initialize Data Structures for Device Functions.

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

    // Step 3: Launch pre-forward kernels.

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

    // Step 4 : Launch forward propagation kernel

    start = chrono::high_resolution_clock::now(); // Log_start

    vector<int> transition_start_indices(num_targets + 1, 0);
    vector<int> cumulative_candidates_per_function(num_targets + 1, 0);

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

    // Array storing the highest likelihood for each candidate
    double *d_best_incoming_weight;
    cudaMalloc(&d_best_incoming_weight, total_num_candidates * sizeof(double));

    vector<double> h_best_incoming_weight(total_num_candidates, -DBL_MAX);
    h_best_incoming_weight[0] = 0.0f; // Start state has a weight of 0.0
    cudaMemcpy(d_best_incoming_weight, h_best_incoming_weight.data(), total_num_candidates * sizeof(double), cudaMemcpyHostToDevice);

    // Array storing the best predessor index for each candidade
    int *d_best_path;
    cudaMalloc(&d_best_path, total_num_candidates * sizeof(int));

    // int h_start_path = -1;  // Start state has no predecessor
    // cudaMemcpy(d_best_path, &h_start_path, sizeof(int), cudaMemcpyHostToDevice);
    vector<int> h_best_path_init(total_num_candidates, -1);
    h_best_path_init[0] = -1; // Start state has no predecessor
    cudaMemcpy(d_best_path, h_best_path_init.data(), total_num_candidates * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize arrays for storing rankings
    // Worst-case: Every candidate is in an unique rank group
    int est_ranking_size = (NUM_CANDIDATES * 2 + 1) * total_num_candidates; // Include '-1' and '-2' markers
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

    // Launch the forward kernel iteratively for each target function

    for (int t = 0; t <= num_targets; ++t) { // Include the transitions from last target function to end state
        int num_prev_candidates = (t == 0) ? 1 : num_candidates_per_function[t - 1];
        int num_current_candidates = (t == num_targets) ? 1 : num_candidates_per_function[t];

        int prev_candidates_start_idx = (t == 0) ? 0 : cumulative_candidates_per_function[t - 1];
        int current_candidates_start_idx = cumulative_candidates_per_function[t];

        int transition_start_idx = transition_start_indices[t];

        // Increase the number of threads per block to its next power of two
        int threads_per_block = num_prev_candidates;
        if (threads_per_block <= 0) {
            threads_per_block = 1;
        } else {
            // Decrement to handle cases where 'num_prev_candidates' is already a power of two
            threads_per_block--;

            // Propagate the highest set bit to all lower bits
            threads_per_block |= threads_per_block >> 1;
            threads_per_block |= threads_per_block >> 2;
            threads_per_block |= threads_per_block >> 4;
            threads_per_block |= threads_per_block >> 8;
            threads_per_block |= threads_per_block >> 16;

            // Increment to get the next power of two
            threads_per_block++;
        }

        int num_blocks = num_current_candidates;

        size_t shared_memory_size = threads_per_block * (sizeof(double) + sizeof(int));

        forwardKernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
            d_transition_likelihoods,
            d_best_incoming_weight,
            d_best_path,
            transition_start_idx,
            prev_candidates_start_idx,
            current_candidates_start_idx,
            num_prev_candidates,
            num_current_candidates,
            num_targets,
            t,
            d_ranking_values,
            d_ranking_offsets,
            d_ambiguous_paths,
            d_ambiguous_paths_offsets,
            d_ambiguous_paths_counts,
            d_group_counts);
        cudaDeviceSynchronize();
    }
    end = chrono::high_resolution_clock::now(); // Log_end
    elapsed = end - start;
    cout << "Forward kernel elapsed time: " << elapsed.count() << " seconds" << endl;

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
    ;

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
    std::string output_dir = parent_str + "RevDecode_results_fine_grained_approach";
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

    cout << "Fine grained approach." << endl;

    for (const auto &input_file_path : input_files) {
        // Extract the base filename without extension
        filesystem::path input_path(input_file_path);
        string filename_stem = input_path.stem().string();

        // Construct the output file path using the original filename
        string output_file_path = output_dir + "/" + filename_stem + "_result_fine_grained_approach.json";

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