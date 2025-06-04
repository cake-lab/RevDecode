// parallel_estimation_approach_forward_kernel.h
#ifndef PARALLEL_ESTIMATION_APPROACH_FORWARD_KERNEL_H
#define PARALLEL_ESTIMATION_APPROACH_FORWARD_KERNEL_H

#include "device_launch_parameters.h"
#include <cmath>

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
 * @kernel properties:
 *		  number of threads = NUM_CANDIDATES
 *        threads per block = NUM_CANDIDATES
 *		  number of blocks = 64

 * @param transition_likelihoods: 1-D array storing all transition likelihoods for candidates across all target functions.
 *                               Each entry represents the transition likelihood from a source candidate to a destination candidate.
 * @param d_best_incoming_weight: Device array holding the best incoming log probability for each candidate state.
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
    int *merging_pair_indices);
#endif