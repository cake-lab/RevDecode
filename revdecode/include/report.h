// report.h
#ifndef REPORT_H
#define REPORT_H

#include "initialize_weight_matrix.h"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using json = nlohmann::json;

/**
 * @brief Generate a report in JSON format with updated rankings for each target function.
 *
 * @param input_file_contents A vector of JSON values representing the input file contents.
 * 							  Each JSON value contains information about a target function and its ranking of candidate functions.
 * @param output_file_path: The file path where the generated report will be saved.
 * @param candidate_function_to_state_machine_map: The mapping between state number and candidate function information.
 * @param new_ranking: A 3D vector, where the outer vector is the target functions, middle vector is the rank groups of candidates, inner vector is the candidates having the same rank.
 */
void report(const vector<json> &input_file_contents,
            const string &output_file_path,
            const vector<CandidateFunction> &candidate_function_to_state_machine_map,
            const vector<vector<vector<int>>> &new_ranking);

#endif