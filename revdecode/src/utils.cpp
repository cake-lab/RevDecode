// utils.cpp

#include "utils.h"

using namespace std;

/**
 * @brief Check the result of a CUDA API call and, if an error occurred, print an error message along with the specific CUDA error string and exits the program.
 *
 * @param err: The CUDA error code returned by a CUDA API call.
 * @param message: A custom error message to be printed if an error occurred.
 */
void checkCudaError(cudaError_t err, const char *message) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << message << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Prints the ranking values for each candidate in each target function.
 *        This function helps in debugging and visualizing the ranking values computed for each candidate state.
 *
 * @param h_ranking_values: A vector storing the ranking values, organized by candidate states.
 *                          Each candidate's rankings are grouped into segments, separated by `-1` as group markers
 *                          and terminated by `-2`.
 * @param h_ranking_offsets: A vector storing offsets for each candidate in `h_ranking_values`.
 *                           Each entry provides the starting index in `h_ranking_values` for the corresponding candidate.
 * @param num_candidates_per_function: A vector storing the number of candidates for each target function.
 *                                      This helps determine how many candidates belong to each target function.
 * @param cumulative_candidates_per_function: A vector storing the cumulative number of candidates across target functions.
 *                                            This is used to calculate the global index for candidates in each target function.
 * @param num_targets: The total number of target functions being processed.
 *                     This determines the loop bounds for iterating over target functions.
 */
void printRankingValues(
    const vector<int> &h_ranking_values,
    const vector<int> &h_ranking_offsets,
    const vector<int> &num_candidates_per_function,
    const vector<int> &cumulative_candidates_per_function,
    int num_targets) {
    cout << "Ranking Values:" << endl;

    for (int t = 0; t < num_targets; ++t) {
        cout << "Target " << t + 1 << ":" << endl;
        int num_candidates_in_target = num_candidates_per_function[t];
        int base_candidate_index = cumulative_candidates_per_function[t]; // Start index for this target's candidates

        for (int c = 0; c < num_candidates_in_target; ++c) {
            int candidate_index = base_candidate_index + c; // Adjusted candidate index
            int start_offset = h_ranking_offsets[candidate_index];

            cout << "  Candidate " << c + 1 << ": [";

            bool first = true;
            for (int i = start_offset; h_ranking_values[i] != -2; ++i) {
                if (h_ranking_values[i] == -1) {
                    cout << " | "; // Separator for ranking groups
                } else {
                    if (!first)
                        cout << ", ";
                    cout << h_ranking_values[i];
                    first = false;
                }
            }
            cout << "]" << endl;
        }
    }
}