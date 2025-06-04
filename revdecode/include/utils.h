// utils.h
#ifndef UTILS_H
#define UTILS_H

#include "device_launch_parameters.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Check the result of a CUDA API call and, if an error occurred, print an error message along with the specific CUDA error string and exits the program.
 *
 * @param err: The CUDA error code returned by a CUDA API call.
 * @param message: A custom error message to be printed if an error occurred.
 */
void checkCudaError(cudaError_t err, const char *message);

/**
 * @brief Helper function to round a double to a specific number of decimal places.
 *
 * @param value: The double value to round.
 * @param precision: The number of decimal places.
 * @return The rounded value.
 */
__device__ __host__ inline double roundToDigits(double value, int precision) {
    double scale = pow(10.0, precision);
    return round(value * scale) / scale;
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
    int num_targets);
#endif