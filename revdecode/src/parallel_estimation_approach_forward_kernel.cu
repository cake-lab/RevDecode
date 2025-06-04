// parallel_estimation_approach_forward_kernel.cu
#include "parallel_estimation_approach_forward_kernel.h"
#include "utils.h"
#include <stdio.h>

#define NUM_CANDIDATES 512 // Total number of candidates per target function (TOP_K + 1 "Unknown" function)

/**
 * @brief This kernel performs the forward propagation across candidate states for each target function, with a phased merging approach.
 *        In phase 0, each thread block independently processes a subset of target functions and performs forward propagation.
 *        Subsequent phases (merging phases) handle merging of boundary functions across pairs of adjacent thread blocks, reducing
 *        the number of active thread blocks by half in each phase. For cases where there are an odd number of blocks, the last block
 *        is carried over without merging and incorporated into the next phase.
 *
 *        Each merging phase computes forward propagation for boundary target functions between two adjacent block segments. If there is
 *        an odd number of blocks in a phase, the last block’s boundary target functions are directly forwarded to the next phase. This
 *        process continues until only one block remains, at which point all boundary target functions have been merged.
 *
 *  @kernel properties:
 *		  number of threads = NUM_CANDIDATES
 *        threads per block = NUM_CANDIDATES
 *		  number of blocks = 64 (change according to device allowed maximum concurrent thread blocks)
 *
 * @param transition_likelihoods: 1-D array storing all transition likelihoods for candidates across all target functions.
 *                               Each entry represents the transition likelihood from a source candidate to a destination candidate.
 * @param d_best_incoming_weight: Device array holding the best incoming probability for each candidate state.
 *                               Each thread will update this array with the maximum likelihood transition it finds.
 * @param d_best_path: Device array holding the index of the best previous candidate (source state) for each candidate.
 *                    Each thread will store the best incoming candidate index for its assigned destination candidate.
 * @param transition_start_indices: Device array of starting indices in `transition_likelihoods` for each target function.
 *                                 This provides the correct index offset in `transition_likelihoods` for each target function.
 * @param cumulative_candidates_per_function: Device array of cumulative sums of candidates up to each target function.
 *                                           This allows each thread to calculate its global candidate index in 'best_incoming_weight'.
 * @param num_targets: The total number of target functions.
 * @param total_num_candidates: The total number of candidates.
 * @param d_ranking_values: Device array to store ranking values for each candidate, indicating the best incoming paths.
 *                         Each thread will store the sorted incoming candidates based on likelihoods.
 * @param d_ranking_offsets: Device array to store offsets in `d_ranking_values` for each candidate's ranking.
 *                          Each thread calculates its offset to store its candidate's ranking values.
 * @param d_ambiguous_paths: Device array to store indices of ambiguous paths for each candidate (paths with equal maximum probability).
 *                          Each thread will populate this array with candidates having maximum likelihood.
 * @param d_ambiguous_paths_offsets: Device array to store starting offsets for each candidate’s ambiguous paths in `d_ambiguous_paths`.
 *                                  This helps each thread locate its portion in the `d_ambiguous_paths` array.
 * @param d_ambiguous_paths_counts: Device array to store counts of ambiguous paths for each candidate.
 *                                 Each thread will store the number of ambiguous paths for its candidate.
 * @param d_group_counts: Device array to store the number of unique probability groups (distinct likelihoods) for each candidate.
 *                       This helps organize candidates with shared or equal likelihoods for ranking purposes.
 * @param block_assignments： Array assigning each target function to a specific thread block in phase 0.
 * @param phase： Current phase of merging. Phase 0 is the initial computation phase, followed by merging phases.
 * @param num_merging_pairs： Number of merging pairs in the current merging phase, calculated as half the number of blocks in the previous phase.
 * @param merging_pair_indices： Array storing the boundary target function indices for each merging pair in the current phase.
 *                             Each merging pair consists of two adjacent segments of boundary target functions.
 */
__launch_bounds__(512)
__global__ void forwardKernel(
    double *transition_likelihoods,
    double *d_best_incoming_weight,
    int *d_best_path,
    int *transition_start_indices,
    int *cumulative_candidates_per_function,
    int num_targets,
    int total_num_candidates,
    int *d_ranking_values,
    int *d_ranking_offsets,
    int *d_num_candidates_per_function,
    int *d_ambiguous_paths,
    int *d_ambiguous_paths_offsets,
    int *d_ambiguous_paths_counts,
    int *d_group_counts,
    int *block_assignments,
    int phase,
    int num_merging_pairs,
    int *merging_pair_indices) {
    int idx = threadIdx.x;
    int block_id = blockIdx.x;

    // Per-thread arrays, allocated in local memory
    double probs[NUM_CANDIDATES];
    int indices[NUM_CANDIDATES];

    // In phase 0, each block processes 'num_targets' / 64 target functions independently
    if (phase == 0) {
        // Determine the range of target functions assigned to the associated block index
        int start_target = -1;
        int end_target = -1;
        for (int i = 0; i < num_targets; ++i) {
            if (block_assignments[i] == block_id) {
                if (start_target == -1) {
                    start_target = i;
                }
                end_target = i;
            }
        }

        // If the block is idle, return control to the main thread on host
        if (start_target == -1) {
            return;
        }

        // Start forward propagation for the assigned range of target functions

        // If this block handles the transitions from the start state
        if (idx == 0 && start_target == 0) {
            d_best_incoming_weight[0] = 0.0f; // Start state has weight 0
            d_best_path[0] = -1;              // Start state has no predecessor
        }
        __syncthreads();

        for (int current_target_function = start_target; current_target_function <= end_target; ++current_target_function) {
            int num_prev_candidates;
            int num_current_candidates = d_num_candidates_per_function[current_target_function];

            if (idx >= num_current_candidates) { // If threads are found idle, skip immediately
                // Do nothing
            } else {
                bool skip_further_computation = false;
                int source_weight_offset;      // Offset of the source candidate in 'd_best_incoming_weight'
                int destination_weight_offset; // Offset of the destination candidate in 'transition_likelihoods'

                // Case 1: First target function in first block processing transitions from start state
                if (current_target_function == 0 && start_target == 0) {
                    num_prev_candidates = 1;
                    source_weight_offset = 0;
                    destination_weight_offset = 0;
                }

                // Case 2: First target function in this block (excluding the first block)
                else if (current_target_function == start_target && start_target != 0) {
                    // Defer this computation to next phase and set 'd_best_incoming_weight' to 0
                    int dest_idx = cumulative_candidates_per_function[current_target_function] + idx;
                    d_best_incoming_weight[dest_idx] = 1;
                    d_best_path[dest_idx] = -1;
                    skip_further_computation = true;
                }

                // Case 3: Other target functions
                else {
                    num_prev_candidates = d_num_candidates_per_function[current_target_function - 1];
                    source_weight_offset = cumulative_candidates_per_function[current_target_function - 1];
                    destination_weight_offset = transition_start_indices[current_target_function];
                }

                if (!skip_further_computation){
                    // Do the forward computation for Cases 1 & 3
                    int destination_candidate_index = cumulative_candidates_per_function[current_target_function] + idx;

                    for (int s = 0; s < num_prev_candidates; ++s) {
                        int source_weight_index = source_weight_offset + s;
                        int destination_weight_index = destination_weight_offset + idx * num_prev_candidates + s;

                        double transition_likelihood = transition_likelihoods[destination_weight_index];

                        double total_likelihood = roundToDigits(d_best_incoming_weight[source_weight_index] + transition_likelihood, 8);

                        // Store the updated total likelihood and its associated source candidate index
                        probs[s] = total_likelihood;
                        indices[s] = source_weight_index;
                    }

                    // Sort the probabilities and corresponding indices in descending order using insertion sort
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

                    // Store the maximum likelihood and best path in global memory
                    d_best_incoming_weight[destination_candidate_index] = probs[0];
                    d_best_path[destination_candidate_index] = indices[0];

                    // Compute group counts
                    int group_count = 1;
                    for (int i = 1; i < num_prev_candidates; ++i) {
                        if (fabs(probs[i] - probs[i - 1]) >= 1e-8) {
                            group_count++;
                        }
                    }
                    d_group_counts[destination_candidate_index] = group_count;

                    // Compute ambiguous paths counts
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

                    int rank_offset = d_ranking_offsets[destination_candidate_index];
                    int amb_offset = d_ambiguous_paths_offsets[destination_candidate_index];

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

                    int amb_idx = amb_offset;
                    for (int i = 0; i < ambiguous_count; ++i) {
                        d_ambiguous_paths[amb_idx++] = indices[i];
                    }
                    d_ambiguous_paths[amb_idx++] = -2; // End of state marker
                }
            }
            // Synchronize threads after each target function
            __syncthreads();
        }

        // If this blocks handles the transitions to the end state
        if (idx == 0 && end_target == num_targets - 1) {
            int last_target_function = num_targets - 1;
            int num_last_candidates = d_num_candidates_per_function[last_target_function];

            int source_weight_offset = cumulative_candidates_per_function[last_target_function];
            int destination_candidate_index = total_num_candidates - 1; // Index of end state

            for (int s = 0; s < num_last_candidates; ++s) {
                int source_weight_index = source_weight_offset + s;

                // Transition likelihood is assumed to be 1.0 from last target function to end state
                double transition_likelihood = 1.0;

                double total_likelihood = roundToDigits(d_best_incoming_weight[source_weight_index] + transition_likelihood, 8);

                probs[s] = total_likelihood;
                indices[s] = source_weight_index;
            }

            // Sort the probabilities and corresponding indices in descending order for last target function
            for (int i = 1; i < num_last_candidates; ++i) {
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

            // Store the ranking information in global memory
            d_best_incoming_weight[destination_candidate_index] = probs[0];
            d_best_path[destination_candidate_index] = indices[0];

            int group_count = 1;
            for (int i = 1; i < num_last_candidates; ++i) {
                if (fabs(probs[i] - probs[i - 1]) >= 1e-8) {
                    group_count++;
                }
            }
            d_group_counts[destination_candidate_index] = group_count;

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

            int rank_offset = d_ranking_offsets[destination_candidate_index];
            int amb_offset = d_ambiguous_paths_offsets[destination_candidate_index];

            // Store rankings in global memory
            int rank_idx = rank_offset;
            for (int i = 0; i < num_last_candidates; ++i) {
                d_ranking_values[rank_idx++] = indices[i];
                // Insert -1 as a marker between different probability groups
                if (i < num_last_candidates - 1 && fabsf(probs[i] - probs[i + 1]) >= 1e-8) {
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
        // print d_best_incoming_weight
        // for (int i = 0; i < total_num_candidates; ++i) {
        //     printf("d_best_incoming_weight[%d] = %f\n", i, d_best_incoming_weight[i]);
        // }
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     for (int i = 0; i < total_num_candidates; ++i) {
        //         printf("d_best_incoming_weight[%d] = %f\n", i, d_best_incoming_weight[i]);
        //     }
        // }
    }

    // Merging phases
    else {
        // The number of merging pairs will be reduced by half in each phase
        // Each merging pair represents two adjacent pairs of <left-boundary, right-boundary> target functions
        if (block_id >= num_merging_pairs) {
            return;
        }

        int merging_pair_index = block_id;

        // Get the indices of the boundary target functions for this merging pair
        int t0 = merging_pair_indices[4 * merging_pair_index];     // Left boundary of first block in previous phase
        int t1 = merging_pair_indices[4 * merging_pair_index + 1]; // Right boundary of first block in previous phase
        int t2 = merging_pair_indices[4 * merging_pair_index + 2]; // Left boundary of second block in previous phase
        int t3 = merging_pair_indices[4 * merging_pair_index + 3]; // Right boundary of second block in previous phase

        // Forward propagation the 4 target functions in sequence
        int target_functions[4] = {t0, t1, t2, t3};

        for (int tf_idx = 0; tf_idx < 4; ++tf_idx) {
            int current_target_function = target_functions[tf_idx];

            // Skip threads that are out of the candidate range for this target function
            int num_current_candidates = d_num_candidates_per_function[current_target_function];
            if (idx >= num_current_candidates) {
                // Do nothing
            } else {
                int destination_candidate_index = cumulative_candidates_per_function[current_target_function] + idx;

                int num_prev_candidates;
                int source_weight_offset; // Offset in 'best_incoming_weight'

                // Do forward propagation computation for candidates in the two adjacent target functions in current phase.
                // The transition likelihoods depend on if this pair of target functions were adjacent in phase 0 (original target functions' sequence)

                // Case 1: First target function (t0), the transition likelihoods of the candidates in it are already set in phase 0 and won't be changed
                if (tf_idx == 0) {
                    // Do nothing
                }

                // Case 2: Non-adjacent target functions in the original target functions order (<t0, t1> and <t2, t3>)
                // The transition likelihoods will use the 'best_incoming_weight' values in the previous phase
                else if (tf_idx % 2 == 1) {
                    num_prev_candidates = d_num_candidates_per_function[target_functions[tf_idx - 1]];
                    source_weight_offset = cumulative_candidates_per_function[target_functions[tf_idx - 1]];

                    // Iterate over previous candidates
                    for (int s = 0; s < num_prev_candidates; ++s) {
                        int source_candidate_index = source_weight_offset + s;

                        double total_likelihood = roundToDigits(d_best_incoming_weight[source_candidate_index] + d_best_incoming_weight[destination_candidate_index], 8);

                        probs[s] = total_likelihood;
                        indices[s] = source_candidate_index;
                    }

                    // Update only 'best_incoming_weight' in global memory
                    double max_total_likelihood = probs[0];
                    for (int i = 1; i < num_prev_candidates; ++i) {
                        if ((probs[i] - max_total_likelihood) >= 1e-8) {
                            max_total_likelihood = probs[i];
                        }
                    }

                    d_best_incoming_weight[destination_candidate_index] = max_total_likelihood;
                }

                // Case 3 : Adjacent target functions in the original target functions order (<t1, t2>)
                // The transition likelihoods will use the values stored in the likelihoods matrix
                else {
                    num_prev_candidates = d_num_candidates_per_function[target_functions[tf_idx - 1]];
                    source_weight_offset = cumulative_candidates_per_function[target_functions[tf_idx - 1]];
                    int destination_weight_offset = transition_start_indices[current_target_function];

                    // Iterate over previous candidates
                    for (int s = 0; s < num_prev_candidates; ++s) {
                        int source_candidate_index = source_weight_offset + s;
                        int transition_index = destination_weight_offset + idx * num_prev_candidates + s;

                        double transition_likelihood = transition_likelihoods[transition_index];

                        double total_likelihood = roundToDigits(d_best_incoming_weight[source_candidate_index] + transition_likelihood, 8);

                        probs[s] = total_likelihood;
                        indices[s] = source_candidate_index;
                    }

                    // Sort the probabilities and corresponding indices in descending order
                    for (int i = 1; i < num_prev_candidates; ++i) {
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

                    // Update ranking values, best paths, and ambiguous paths in global memory
                    d_best_incoming_weight[destination_candidate_index] = probs[0];
                    d_best_path[destination_candidate_index] = indices[0];

                    int group_count = 1;
                    for (int i = 1; i < num_prev_candidates; ++i) {
                        if (fabs(probs[i] - probs[i - 1]) >= 1e-8) {
                            group_count++;
                        }
                    }
                    d_group_counts[destination_candidate_index] = group_count;

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

                    int rank_offset = d_ranking_offsets[destination_candidate_index];
                    int rank_idx = rank_offset;
                    for (int i = 0; i < num_prev_candidates; ++i) {
                        d_ranking_values[rank_idx++] = indices[i];
                        if (i < num_prev_candidates - 1 && fabs(probs[i] - probs[i + 1]) >= 1e-8) {
                            d_ranking_values[rank_idx++] = -1;
                        }
                    }
                    d_ranking_values[rank_idx++] = -1; // End marker
                    d_ranking_values[rank_idx++] = -2; // End of state marker

                    int amb_offset = d_ambiguous_paths_offsets[destination_candidate_index];
                    int amb_idx = amb_offset;
                    for (int i = 0; i < ambiguous_count; ++i) {
                        d_ambiguous_paths[amb_idx++] = indices[i];
                    }
                    d_ambiguous_paths[amb_idx++] = -2; // End of state marker
                }
            }

            // Synchronize threads after each target function
            __syncthreads();
        }
    }
}
