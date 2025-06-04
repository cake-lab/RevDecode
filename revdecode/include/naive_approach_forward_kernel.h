// naive_approach_forward_kernel.h
#ifndef NAIVE_APPROACH_FORWARD_KERNEL_H
#define NAIVE_APPROACH_FORWARD_KERNEL_H

#include "device_launch_parameters.h"
#include <cmath>

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
    int *d_group_counts);

#endif