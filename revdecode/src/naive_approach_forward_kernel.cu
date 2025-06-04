// naive_approach_forward_kernel.cu
#include "naive_approach_forward_kernel.h"
#include "utils.h"

#define NUM_CANDIDATES 512 // Total number of candidates per target function (TOP_K + 1 "Unknown" function)

/**
 * @brief Perform the forward pass of the Viterbi algorithm for each target function.
 *		  This kernel is designed to process each target function in the forward pass of the Viterbi algorithm.
 *		  Each thread is responsible for one candidate in the current target function, calculating and storing
 *		  the best incoming path, and ranking values.
 *
 * @kernel properties:
 *		  number of threads = NUM_CANDIDATES
 *        threads per block = NUM_CANDIDATES
 *		  number of blocks = 1
 *
 * @param transition_likelihoods: 1-D array storing all transition likelihoods for candidates across all target functions.
 *                               Each entry represents the transition likelihood from a source candidate to a destination candidate.
 * @param d_best_incoming_weight: Device array holding the best incoming weight for each candidate state.
 *                               Each thread will update this array with the maximum likelihood transition it finds.
 * @param d_best_path: Device array holding the index of the best previous candidate (source state) for each candidate.
 *                    Each thread will store the best incoming candidate index for its assigned destination candidate.
 * @param transition_start_indices: Device array of starting indices in `transition_likelihoods` for each target function.
 *                                 This provides the correct index offset in `transition_likelihoods` for each target function.
 * @param cumulative_candidates_per_function: Device array of cumulative sums of candidates up to each target function.
 *                                           This allows each thread to calculate its global candidate index in 'best_incoming_weight'.
 * @param num_target_functions: The total number of target functions.
 * @param total_num_candidates: The total number of candidates.
 * @param d_ranking_values: Device array to store ranking values for each candidate, indicating the best incoming paths.
 *                         Each thread will store the sorted incoming candidates based on likelihoods.
 * @param d_ranking_offsets: Device array to store offsets in `d_ranking_values` for each candidate's ranking.
 *                          Each thread calculates its offset to store its candidate's ranking values.
 * @param num_candidates_per_function： Array storing the number of candidates for each target function.
 * @param d_ambiguous_paths: Device array to store indices of ambiguous paths for each candidate (paths with equal maximum probability).
 *                          Each thread will populate this array with candidates having maximum likelihood.
 * @param d_ambiguous_paths_offsets: Device array to store starting offsets for each candidate’s ambiguous paths in `d_ambiguous_paths`.
 *                                  This helps each thread locate its portion in the `d_ambiguous_paths` array.
 * @param d_ambiguous_paths_counts: Device array to store counts of ambiguous paths for each candidate.
 *                                 Each thread will store the number of ambiguous paths for its candidate.
 * @param d_group_counts: Device array to store the number of unique probability groups (distinct likelihoods) for each candidate.
 *                       This helps organize candidates with shared or equal likelihoods for ranking purposes.
 */

__global__ void forwardKernel(
    double *transition_likelihoods,
    double *d_best_incoming_weight,
    int *d_best_path,
    int *transition_start_indices,
    int *cumulative_candidates_per_function,
    int num_target_functions,
    int total_num_candidates,
    int *d_ranking_values,
    int *d_ranking_offsets,
    int *num_candidates_per_function,
    int *d_ambiguous_paths,
    int *d_ambiguous_paths_offsets,
    int *d_ambiguous_paths_counts,
    int *d_group_counts) {
    int idx = threadIdx.x;

    // Per-thread arrays, allocated in local memory
    double probs[NUM_CANDIDATES];
    int indices[NUM_CANDIDATES];

    // Initialize d_best_incoming_weight and d_best_path for the start state
    if (idx == 0) {
        d_best_incoming_weight[0] = 0.0f; // Start state has weight 0
        d_best_path[0] = -1;              // Start state has no predecessor
    }
    __syncthreads();

    // Iterate each target function, sync threads every time after finishing computation for a target function
    for (int current_target_function = 0; current_target_function < num_target_functions; ++current_target_function) {
        int num_prev_candidates;

        // Skip threads that are out of the candidate range for this target function
        int num_current_candidates = num_candidates_per_function[current_target_function];
        if (idx >= num_current_candidates) {
            // Do nothing
        }

        else {
            int source_weight_offset;      // Offset of the source candidate in 'd_best_incoming_weight'
            int destination_weight_offset; // Offset of the destination candidate in 'transition_likelihoods'

            if (current_target_function == 0) { // Transitions from start state
                num_prev_candidates = 1;
                source_weight_offset = 0;
                destination_weight_offset = 0;
            } else {
                num_prev_candidates = num_candidates_per_function[current_target_function - 1];
                source_weight_offset = cumulative_candidates_per_function[current_target_function - 1];
                destination_weight_offset = transition_start_indices[current_target_function];
            }

            // Destination candidate index in the d_best_incoming_weights, d_best_paths we are going to store
            int destination_candidate_index = cumulative_candidates_per_function[current_target_function] + idx;

            // Collect transition likelihoods and compute total likelihoods
            for (int s = 0; s < num_prev_candidates; ++s) {
                int source_weight_index = source_weight_offset + s;
                int destination_weight_index = destination_weight_offset + idx * num_prev_candidates + s;

                double transition_likelihood = transition_likelihoods[destination_weight_index];

                double total_likelihood;

                if (current_target_function == 0) {
                    total_likelihood = transition_likelihood;
                } else {
                    total_likelihood = roundToDigits(d_best_incoming_weight[source_weight_index] + transition_likelihood, 8);
                }

                // Store the updated transition likelihoods and its associated source candidate index to the current destination candidate
                probs[s] = total_likelihood;
                indices[s] = source_weight_index;
            }

            // Sort the probabilities and corresponding indices in descending order, using insertion sort
            for (int i = 1; i < num_prev_candidates; ++i) {
                double key_prob = probs[i];
                int key_idx = indices[i];
                int j = i - 1;

                while (j >= 0 && (probs[j] - key_prob) < 1e-8) { // Swap when probs[j] is less than or approximately equal to key_prob
                    probs[j + 1] = probs[j];
                    indices[j + 1] = indices[j];
                    j--;
                }
                probs[j + 1] = key_prob;
                indices[j + 1] = key_idx;
            }

            // Store the values associated with the candidate computed by each thread in global memory

            // Store the maximum likelihood and best path in global memory
            d_best_incoming_weight[destination_candidate_index] = probs[0];
            d_best_path[destination_candidate_index] = indices[0];

            // Store ranking groups in global memory
            int group_count = 1;
            for (int i = 1; i < num_prev_candidates; ++i) {
                if (fabs(probs[i] - probs[i - 1]) >= 1e-8) {
                    group_count++;
                }
            }
            d_group_counts[destination_candidate_index] = group_count;

            // Store ambiguous paths which have the same maximum likelihood in global memory
            int ambiguous_count = 1;
            double max_prob = probs[0];
            for (int i = 1; i < num_prev_candidates; ++i) {
                if (fabs(probs[i] - max_prob) < 1e-8) {
                    ambiguous_count++;
                } else {
                    break;
                }
            }
            d_ambiguous_paths_counts[destination_candidate_index] = ambiguous_count;

            // Retrieve the offsets for the current destination candidate in 'd_ranking_values'
            int rank_offset = d_ranking_offsets[destination_candidate_index];
            int amb_offset = d_ambiguous_paths_offsets[destination_candidate_index];

            // Store rankings in global memory
            int rank_idx = rank_offset;
            for (int i = 0; i < num_prev_candidates; ++i) {
                d_ranking_values[rank_idx++] = indices[i];
                // Insert -1 as a marker between different probability groups
                if (i < num_prev_candidates - 1 && fabs(probs[i] - probs[i + 1]) >= 1e-8) {
                    d_ranking_values[rank_idx++] = -1;
                }
            }
            d_ranking_values[rank_idx++] = -1; // End marker
            d_ranking_values[rank_idx++] = -2; // End of state marker

            // Populate the ambiguous paths array
            int amb_idx = amb_offset;
            for (int i = 0; i < ambiguous_count; ++i) {
                d_ambiguous_paths[amb_idx++] = indices[i];
            }
            d_ambiguous_paths[amb_idx++] = -2; // End of state mark
        }
        // Synchronize threads after each target function
        __syncthreads();
    }

    // Reuse thread 0 to handle the transitions from last target function to end state
    if (idx == 0) {
        int last_target_function = num_target_functions - 1;
        int num_last_candidates = num_candidates_per_function[last_target_function];

        int source_weight_offset = cumulative_candidates_per_function[last_target_function];
        int destination_candidate_index = total_num_candidates - 1; // Index of end state

        for (int s = 0; s < num_last_candidates; ++s) {
            int source_weight_index = source_weight_offset + s;

            // Transition likelihood is assumed to be 1.0
            double transition_likelihood = 1.0;

            double total_likelihood = roundToDigits(d_best_incoming_weight[source_weight_index] + transition_likelihood, 8);

            probs[s] = total_likelihood;
            indices[s] = source_weight_index;
        }

        // Sort the probabilities and corresponding indices in descending order
        for (int i = 1; i < num_last_candidates; ++i) {
            double key_prob = probs[i];
            int key_idx = indices[i];
            int j = i - 1;

            while (j >= 0 && (probs[j] - key_prob) < 1e-8) {
                probs[j + 1] = probs[j];
                indices[j + 1] = indices[j];
                j--;
            }
            probs[j + 1] = key_prob;
            indices[j + 1] = key_idx;
        }

        // Store the maximum likelihood and the corresponding source candidate index for the end state
        d_best_incoming_weight[destination_candidate_index] = probs[0];
        d_best_path[destination_candidate_index] = indices[0];

        // Compute group counts
        int group_count = 1;
        for (int i = 1; i < num_last_candidates; ++i) {
            if (fabs(probs[i] - probs[i - 1]) >= 1e-8) {
                group_count++;
            }
        }
        d_group_counts[destination_candidate_index] = group_count;

        // Compute ambiguous paths counts
        int ambiguous_count = 1;
        double max_prob = probs[0];
        for (int i = 1; i < num_last_candidates; ++i) {
            if (fabs(probs[i] - max_prob) < 1e-8) {
                ambiguous_count++;
            } else {
                break;
            }
        }
        d_ambiguous_paths_counts[destination_candidate_index] = ambiguous_count;

        // Retrieve offsets
        int rank_offset = d_ranking_offsets[destination_candidate_index];
        int amb_offset = d_ambiguous_paths_offsets[destination_candidate_index];

        // Store rankings in global memory
        int rank_idx = rank_offset;
        for (int i = 0; i < num_last_candidates; ++i) {
            d_ranking_values[rank_idx++] = indices[i];
            // Insert -1 as a marker between different probability groups
            if (i < num_last_candidates - 1 && fabs(probs[i] - probs[i + 1]) >= 1e-8) {
                d_ranking_values[rank_idx++] = -1;
            }
        }
        d_ranking_values[rank_idx++] = -1; // End marker
        d_ranking_values[rank_idx++] = -2; // End of state marker

        // Populate the ambiguous paths array
        int amb_idx = amb_offset;
        for (int i = 0; i < ambiguous_count; ++i) {
            d_ambiguous_paths[amb_idx++] = indices[i];
        }
        d_ambiguous_paths[amb_idx++] = -2; // End of state marker
    }
}