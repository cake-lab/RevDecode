// backtrack.cpp
#include "backtrack.h"
#include <iostream>

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
vector<vector<int>> merge_ranking(const vector<vector<vector<int>>> &raw_ranking) {
    vector<vector<int>> combined;

    for (size_t i = 0; i < raw_ranking.size(); i++) {
        for (size_t j = 0; j < raw_ranking[i].size(); ++j) {
            if (combined.size() <= j) {
                combined.push_back({});
            }
            combined[j].insert(combined[j].end(), raw_ranking[i][j].begin(), raw_ranking[i][j].end());
        }
    }

    vector<vector<int>> merged_ranking;

    // Create a set to keep track of seen items. This is to make sure if one state appears in the higher ranking, it won't appear again in the lower ranking
    unordered_set<int> seen;

    for (const auto &sublist : combined) {
        vector<int> new_sublist;
        for (const auto &item : sublist) {
            // Add item if it is not in seen and it is not None
            if (seen.find(item) == seen.end()) {
                new_sublist.push_back(item);
                seen.insert(item);
            }
        }

        if (!new_sublist.empty()) {
            merged_ranking.push_back(new_sublist);
        }
    }

    return merged_ranking;
}

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
    int num_targets) {
    // 'ambiguous_states_per_target' stores the set of candidates on the ambiguous path for each target function
    // Candidates on the ambiguous paths are the rank-1 candidates having the same ranking values
    vector<unordered_set<int>> ambiguous_states_per_target(num_targets); // Include end state and exlcude the first target function

    // Construct the ambiguous predecessor states starting from the end state
    int end_state_idx = total_num_candidates - 1;
    int offset = h_ambiguous_paths_offsets[end_state_idx];
    int count = h_ambiguous_paths_counts[end_state_idx];
    // print contents of h_ranking_values
    // cout << "h_ranking_values size: " << h_ranking_values.size() << endl;
    // cout << "h_ranking_values: ";
    // for (int i = 0; i < h_ranking_values.size(); ++i) {
    //     cout << h_ranking_values[i] << " ";
    // }
    // cout << endl;

    for (int i = 0; i < count; ++i) {
        int p = h_ambiguous_paths[offset + i];

        if (p != -2) { // We used '-2' to mark the end of ambiguous paths for a candidate
            ambiguous_states_per_target[num_targets - 1].insert(p);
        }
    }

    // Now backtrack to get the ambiguous predecessor states for each target function, stopping at the 2nd target function
    for (int t = num_targets - 1; t >= 1; --t) {
        for (int s : ambiguous_states_per_target[t]) {
            // Get ambiguous predecessor states of s
            int offset_s = h_ambiguous_paths_offsets[s];
            int count_s = h_ambiguous_paths_counts[s];

            for (int i = 0; i < count_s; ++i) {
                int p = h_ambiguous_paths[offset_s + i];
                if (p != -2) {
                    ambiguous_states_per_target[t - 1].insert(p);
                }
            }
        }
    }

    vector<vector<vector<int>>> new_ranking(num_targets);

    // For the last target function (t = num_targets - 1), directly use the rankings of the end state
    int t_last = num_targets - 1;
    vector<vector<int>> state_rankings;
    vector<int> rank_group;

    int rank_index = h_ranking_offsets[end_state_idx];
    // cout << "Bug starts here." << endl;
    // cout << "rank_index: " << rank_index << endl;
    // cout << "h_ranking_values[rank_index]: " << h_ranking_values[rank_index] << endl;

    while (h_ranking_values[rank_index] != -2) {  // End of current candidate's ranking groups
        if (h_ranking_values[rank_index] == -1) { // Next ranking group
            if (!rank_group.empty()) {
                state_rankings.push_back(rank_group);
                rank_group.clear();
            }
        } else {
            rank_group.push_back(h_ranking_values[rank_index]);
        }
        ++rank_index;
    }

    if (!rank_group.empty()) {
        state_rankings.push_back(rank_group);
    }

    new_ranking[t_last] = state_rankings;

    // For t from num_targets - 1 down to 1, use ambiguous paths to merge rankings for previous target function
    for (int t = num_targets - 1; t >= 1; --t) {
        unordered_set<int> &ambiguous_states = ambiguous_states_per_target[t];
        vector<vector<vector<int>>> raw_ranking;

        // Collect rankings of ambiguous candidates for this target function
        for (int s : ambiguous_states) {
            vector<vector<int>> state_rankings;
            vector<int> rank_group;
            int rank_index = h_ranking_offsets[s];

            while (h_ranking_values[rank_index] != -2) {
                if (h_ranking_values[rank_index] == -1) {
                    if (!rank_group.empty()) {
                        state_rankings.push_back(rank_group);
                        rank_group.clear();
                    }
                }

                else {
                    rank_group.push_back(h_ranking_values[rank_index]);
                }
                ++rank_index;
            }

            if (!rank_group.empty()) {
                state_rankings.push_back(rank_group);
            }

            raw_ranking.push_back(state_rankings);
        }

        // Merge the rankings of these ambiguous states
        vector<vector<int>> merged_ranking = merge_ranking(raw_ranking);

        // Store the merged ranking as the new ranking for target function t - 1
        new_ranking[t - 1] = merged_ranking;
    }

    return new_ranking;
}
