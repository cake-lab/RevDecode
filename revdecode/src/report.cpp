// report.cpp
#include "report.h"
#include "update_ranking_scheme.h"
#include <iostream>

using namespace std;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

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
            const vector<vector<vector<int>>> &new_ranking) {
    ordered_json LibDec0de_report; // Create a JSON value to store the final report
    int report_index = 0;

    for (const auto &target_function : input_file_contents) {
        ordered_json new_information;
        new_information["Target_function"] = target_function["Target_function"];
        new_information["Corpus_version"] = target_function["Corpus_version"];
        new_information["self_confidence_score"] = target_function["self_confidence_score"];
        new_information["self_confidence_factor"] = target_function["self_confidence_factor"];
        new_information["unknown_confidence_score"] = target_function["unknown_confidence_score"];
        new_information["unknown_confidence_factor"] = target_function["unknown_confidence_factor"];

        vector<ordered_json> old_ranking(target_function["ranking"].begin(), target_function["ranking"].end());
        new_information["ranking"] = ordered_json::array();

        if (report_index < static_cast<int>(new_ranking.size())) {
            // Build a map for old_ranking for efficient lookup
            unordered_map<string, ordered_json> old_ranking_map;
            for (const auto &candidate : old_ranking) {
                string key = candidate["library_name"].get<string>() + "|" +
                             candidate["library_version"].get<string>() + "|" +
                             candidate["compiler_option"].get<string>() + "|" +
                             candidate["function_name"].get<string>() + "|" +
                             candidate["compiler_unit"].get<string>();
                old_ranking_map[key] = candidate;
            }

            int rank_number = 1; // Start ranking from 1

            // Loop through each rank group in the new ranking for the current target function.
            for (const auto &rank_group : new_ranking[report_index]) {
                for (const auto &state : rank_group) {
                    if (state >= candidate_function_to_state_machine_map.size()) {
                        std::cerr << "ERROR: State index " << state
                                  << " is out of bounds for candidate_function_to_state_machine_map (size = "
                                  << candidate_function_to_state_machine_map.size() << ").\n";
                        continue;
                    }
                    CandidateFunction candidate_info = candidate_function_to_state_machine_map[state];

                    // Create a unique key for the candidate
                    string key = candidate_info.library_name + "|" +
                                 candidate_info.library_version + "|" +
                                 candidate_info.compiler_option + "|" +
                                 candidate_info.function_name + "|" +
                                 candidate_info.compiler_unit;

                    // Remove the candidate from old_ranking_map if it exists
                    auto it = old_ranking_map.find(key);
                    ordered_json candidate_json;

                    if (it != old_ranking_map.end()) {
                        candidate_json = it->second;
                        old_ranking_map.erase(it);                                // Remove to avoid duplicates
                        candidate_json["original_rank"] = candidate_json["rank"]; // Set original_rank
                    } else {
                        // Candidate not found in old_ranking, create new entry
                        candidate_json = {
                            {"library_name", candidate_info.library_name},
                            {"library_version", candidate_info.library_version},
                            {"compiler_option", candidate_info.compiler_option},
                            {"function_name", candidate_info.function_name},
                            {"compiler_unit", candidate_info.compiler_unit},
                            {"similarity_score", candidate_info.similarity_score},
                            {"confidence_factor", candidate_info.emission_probability},
                            {"uniqueness_score", candidate_info.uniqueness_score},
                            {"original_rank", candidate_info.rank}};
                    }

                    candidate_json["rank"] = rank_number;
                    new_information["ranking"].push_back(candidate_json);
                }
                rank_number++;
            }

            // Append the rest of the old rankings to the end of the updated new rankings
            if (!old_ranking_map.empty()) {
                // Collect remaining candidates from the map
                vector<ordered_json> remaining_candidates;
                for (auto &[key, candidate] : old_ranking_map) {
                    candidate["original_rank"] = candidate["rank"];
                    candidate["rank"] = rank_number++;
                    remaining_candidates.push_back(candidate);
                }

                // Sort remaining candidates
                sort(remaining_candidates.begin(), remaining_candidates.end(), [](const ordered_json &a, const ordered_json &b) {
                    return a["original_rank"] < b["original_rank"];
                });

                for (auto &candidate : remaining_candidates) {
                    new_information["ranking"].push_back(candidate);
                }
            }

            // No need to sort `new_information["ranking"]` again by rank, as we've assigned ranks sequentially.

            // Handle ranking ties if necessary
            auto ranking_vec = json_to_vector_map(new_information["ranking"]);
            auto updated_ranking_vec = update_ranking_for_one_target_function(ranking_vec);
            new_information["ranking"] = vector_map_to_json(updated_ranking_vec);

            LibDec0de_report.push_back(new_information);
            report_index++;
        }
    }

    ofstream output_file(output_file_path);
    output_file << LibDec0de_report.dump(4); // Indentation level (in spaces) for nested elements
}
