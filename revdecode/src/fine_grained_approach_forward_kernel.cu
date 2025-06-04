// fine_grained_approach_forward_kernel.cu
#include "fine_grained_approach_forward_kernel.h"
#include <stdio.h>
#include <utils.h>

/**
 * @brief This kernel does forward propagation in the Viterbi algorithm with bitonic sort for ranking candidates.
 *		   Each thread in a block handles transitions from one previous candidate to a single current candidate.
 *		   After processing, the kernel updates the best likelihood and path information for each current candidate.
 *
 * @kernel properties:
 *			total number of threads: number of transitions between two target functions
 *			total number of blocks: number of candidates in the current target function
 *			threads per block: number of transitions from the previous target function to a candidate in the current target function
 *
 * @param transition_likelihoods: 1-D array of transition likelihoods between candidate states.
 * @param d_best_incoming_weight: Device array to store the best incoming likelihood for each candidate.
 * @param d_best_path: Device array to store the best predecessor index for each candidate.
 * @param transition_start_idx: Start index of the transition likelihoods for the current target function in the 1-D array.
 * @param prev_candidate_start_idx: Start index of previous candidates for the current target function.
 * @param current_candidates_start_idx: Start index of current candidates for the current target function.
 * @param num_prev_candidates: Number of candidates in the previous target function.
 * @param num_current_candidates: Number of candidates in the current target function.
 * @param num_target_functions: Total number of target functions being processed.
 * @param current_target_function: Index of the current target function being processed (0-based).
 * @param d_ranking_values: Device array to store ranking values for each candidate.
 *                           Each candidate's rankings are stored in segments marked with `-1` as separators.
 * @param d_ranking_offsets: Device array storing offsets in `d_ranking_values` for each candidate's rankings.
 * @param d_ambiguous_paths: Device array to store indices of ambiguous paths for each candidate.
 *                           Each candidate's ambiguous paths are stored in segments marked with `-2` as separators.
 * @param d_ambiguous_paths_offsets: Device array storing offsets in `d_ambiguous_paths` for each candidate's ambiguous paths.
 * @param d_ambiguous_paths_counts: Device array to store counts of ambiguous paths for each candidate.
 * @param d_group_counts: Device array to store the number of distinct likelihood groups for each candidate.
 */
__global__ void forwardKernel(
    double *transition_likelihoods,
    double *d_best_incoming_weight,
    int *d_best_path,
    int transition_start_idx,
    int prev_candidate_start_idx,
    int current_candidates_start_idx,
    int num_prev_candidates,
    int num_current_candidates,
    int num_target_functions,
    int current_target_function,
    int *d_ranking_values,
    int *d_ranking_offsets,
    int *d_ambiguous_paths,
    int *d_ambiguous_paths_offsets,
    int *d_ambiguous_paths_counts,
    int *d_group_counts) {
    // Each block handles all transitions to one candidate
    int dst_candidate_idx = blockIdx.x; // Local candidate index in the current target function

    int thread_idx = threadIdx.x; // Each thread handles one transition

    if (dst_candidate_idx >= num_current_candidates) { // If a block found idle, immediately skip it
        return;
    }

    // Determine if we are processing the end state
    bool is_end_state = (current_target_function == num_target_functions);

    // Global index for the destination candidate
    int dst_candidate_global_idx = current_candidates_start_idx + dst_candidate_idx;

    extern __shared__ double shared_mem[];

    int shared_size = blockDim.x; // Number of threads of this block determined on host side

    double *probs = shared_mem;
    int *indices = (int *)&probs[shared_size];

    // Initialize shared memory.
    // We store the likelihoods and associated indices in shared memory, and padded the size to power of 2 for the bitonic sorting in later stage
    if (thread_idx < num_prev_candidates) {
        int src_candidate_idx = thread_idx;
        int src_candidate_global_idx = prev_candidate_start_idx + src_candidate_idx;

        int transition_idx = is_end_state ? -1 : transition_start_idx + dst_candidate_idx * num_prev_candidates + src_candidate_idx;

        // Retrieve transition likelihood
        double transition_likelihood = is_end_state ? 1.0 : transition_likelihoods[transition_idx];

        double total_likelihood = roundToDigits(d_best_incoming_weight[src_candidate_global_idx] + transition_likelihood, 8);

        probs[thread_idx] = total_likelihood;
        indices[thread_idx] = src_candidate_global_idx;
    } else {
        // Padding
        probs[thread_idx] = -INFINITY;
        indices[thread_idx] = -1;
    }

    __syncthreads();

    /**
     *
     * Stages of Bitonic Sort:
     * 1. Initialization:
     *    - Ensure the array size is a power of 2 by padding it with `-INFINITY`.
     *    - Each thread block will handle one part of the array (shared memory is used for efficiency).
     *
     * 2. Bitonic Sequence Formation:
     *    - Build smaller bitonic sequences by merging sub-arrays of size 1, 2, 4, ..., up to the size of the array.
     *    - Alternate between ascending and descending orders when merging sub-sequences to form bitonic sequences.
     *
     * 3. Bitonic Merge:
     *    - Recursively compare and swap elements in the bitonic sequence to ensure the entire array is sorted.
     *    - Compare elements separated by a specific distance and swap them based on the required order.
     *    - Continue dividing and merging sub-sequences until the final sorted sequence is achieved.
     *
     * Steps in the Kernel:
     * 1. Padding Check:
     *    - Before starting the sort, calculate the number of threads needed (next power of 2).
     *    - Initialize padded elements to avoid incorrect results during comparisons.
     *
     * 2. Sorting Loop:
     *    - Use a nested loop to implement the iterative version of bitonic sort:
     *      - Outer Loop (size `k`): Controls the size of the sub-sequences being processed.
     *      - Inner Loop (step `j`): Controls the distance between elements being compared.
     *
     * 3. Comparison and Swap:
     *    - Compare elements using XOR indexing (`paired_idx = thread_idx ^ j`).
     *
     * 4. Final Extraction:
     *    - Ignore padding values when writing the final sorted array.
     *    - Output only the valid elements corresponding to the original array size.
     */
    int size = shared_size;

    for (int k = 2; k <= size; k <<= 1) {      // Ensure the paired index is within bounds
        for (int j = k >> 1; j > 0; j >>= 1) { // // Avoid redundant comparisons
            int paired_idx = thread_idx ^ j;

            if (paired_idx < size && paired_idx > thread_idx) { // Ensure the paired index is within bounds and avoid redundant comparisons
                // Determine the sorting direction
                bool descending = ((thread_idx & k) == 0);

                // Swap when probs[thread_idx] is less than or approximately equal to probs[paired_idx], which is correct for descending order.
                // Conversely, if order is ascending, swap if probs[paired_idx] is less than or approximately equal to probs[thread_idx].
                if (((probs[thread_idx] - probs[paired_idx]) < 1e-8 && descending) ||
                    ((probs[paired_idx] - probs[thread_idx]) < 1e-8 && !descending)) {
                    double temp_prob = probs[thread_idx];
                    probs[thread_idx] = probs[paired_idx];
                    probs[paired_idx] = temp_prob;

                    int temp_idx = indices[thread_idx];
                    indices[thread_idx] = indices[paired_idx];
                    indices[paired_idx] = temp_idx;
                }
            }
            __syncthreads();
        }
    }

    // After sorting, proceed to write the outputs

    __syncthreads();

    if (thread_idx == 0) {
        // Process valid entries from index 0
        if (indices[0] != -1) {
            d_best_incoming_weight[dst_candidate_global_idx] = probs[0];
            d_best_path[dst_candidate_global_idx] = indices[0];

            // Count ambiguous paths
            int ambiguous_count = 1;
            double max_prob = probs[0];
            for (int i = 1; i < num_prev_candidates; ++i) {
                if (indices[i] == -1 || fabs(probs[i] - max_prob) >= 1e-8) {
                    break;
                }
                ambiguous_count++;
            }
            d_ambiguous_paths_counts[dst_candidate_global_idx] = ambiguous_count;

            // Store ambiguous paths
            int ambiguous_offset = d_ambiguous_paths_offsets[dst_candidate_global_idx];
            for (int i = 0; i < ambiguous_count; ++i) {
                d_ambiguous_paths[ambiguous_offset + i] = indices[i];
            }
            d_ambiguous_paths[ambiguous_offset + ambiguous_count] = -2; // End marker

            // Count group counts
            int group_count = 1;
            for (int i = 1; i < num_prev_candidates; ++i) {
                if (indices[i] == -1) {
                    break;
                }
                if (fabs(probs[i] - probs[i - 1]) >= 1e-8) {
                    group_count++;
                }
            }
            d_group_counts[dst_candidate_global_idx] = group_count;

            // Store ranking values
            int rank_offset = d_ranking_offsets[dst_candidate_global_idx];
            int rank_idx = rank_offset;
            for (int i = 0; i < num_prev_candidates && indices[i] != -1; ++i) {
                d_ranking_values[rank_idx++] = indices[i];
                if (i < num_prev_candidates - 1 && indices[i + 1] != -1 && (probs[i] - probs[i + 1]) >= 1e-8) {
                    d_ranking_values[rank_idx++] = -1; // Group separator
                }
            }
            d_ranking_values[rank_idx++] = -1; // End marker
            d_ranking_values[rank_idx] = -2;   // End of candidate marker
        }
    }
}
