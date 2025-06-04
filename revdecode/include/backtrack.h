// backtrack.h
#ifndef BACKTRACK_H
#define BACKTRACK_H

#include <unordered_set>
#include <vector>
using namespace std;

/**
 * @brief If a target function has more than one rank-1 candidates, merge their rank values
 * 		  Example:
 *  		Before merging:		[
 *								[[1, 2], [3, 4], [5, 6]],
 * 								[[1, 2], [3, 4, 5, 6]],
 *								[[1, 3], [2, 4], [5, 6]]
 *					    		]
 * 			After merging:
 *								[[1, 2, 3], [4, 5, 6]]
 * @param raw_ranking: A 3D vector representing the new ranking. The outer vector corresponds to current target function's rank-1 candidates,
 * 					   the middle vector corresponds to rank groups, and the inner vector corresponds to the previous target function's candidates having the same rank.
 */
vector<vector<int>> merge_ranking(const vector<vector<vector<int>>> &raw_ranking);

/**
 * @brief Perform backtracking to construct the optimal ranking path.
 *		  This function reconstructs the best path from the `h_best_path` array, collects the ranking groups
 *        for each state along the best path, and merges these groups to produce the final `new_ranking` structure.
 *
 * @param h_ranking_values: A vector storing the ranking values, organized by candidate states.
 *                         Each state has associated ranking groups with candidates ranked by likelihood.
 *                         Markers -1 and -2 are used to separate ranking levels and states, respectively.
 * @param h_ranking_offsets: A vector storing offsets for each candidate in `h_ranking_values`.
 *                          Each entry in `h_ranking_offsets` provides the starting index in
 *                          `h_ranking_values` for the corresponding candidate state.
 * @param total_num_candidates: The total number of candidates (including start and end states).
 *                             This defines the size of `h_best_path`, `h_ranking_offsets`, and `h_ranking_values`.
 * @param d_ambiguous_paths: Device array to store indices of ambiguous paths for each candidate (paths with equal maximum probability).
 *                          Each thread will populate this array with candidates having maximum likelihood.
 * @param d_ambiguous_paths_offsets: Device array to store starting offsets for each candidate’s ambiguous paths in `d_ambiguous_paths`.
 *                                  This helps each thread locate its portion in the `d_ambiguous_paths` array.
 * @param d_ambiguous_paths_counts: Device array to store counts of ambiguous paths for each candidate.
 *                                 Each thread will store the number of ambiguous paths for its candidate.
 * @param d_group_counts: Device array to store the number of unique probability groups (distinct likelihoods) for each candidate.
 *                       This helps organize candidates with shared or equal likelihoods for ranking purposes.
 *
 * @return A 3D vector representing the new ranking path along the best path for each target function.
 *         Each entry in the outermost vector represents a target function, containing rank groups for candidates.
 *         Example structure:
 *         - [[ [1, 2, 3], [4, 5] ],  // Target function 1 with rank groups
 *            [ [6, 7], [8] ],        // Target function 2 with rank groups
 *            ...]
 */

vector<vector<vector<int>>> backtrack(
    const vector<int> &h_ranking_values,
    const vector<int> &h_ranking_offsets,
    const vector<int> &h_ambiguous_paths,
    const vector<int> &h_ambiguous_paths_offsets,
    const vector<int> &h_ambiguous_paths_counts,
    int total_num_candidates,
    int num_targets);
#endif