// fine_grained_approach_forward_kernel.h
#ifndef FINE_GRAINED_APPROACH_FORWARD_KERNEL_H
#define FINE_GRAINED_APPROACH_FORWARD_KERNEL_H

#include "device_launch_parameters.h"
#include <cmath>

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
    int *d_group_counts);

#endif