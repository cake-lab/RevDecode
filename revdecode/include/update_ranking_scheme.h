// update_ranking_scheme.h
#ifndef UPDATE_RANKING_SCHEME_H
#define UPDATE_RANKING_SCHEME_H

#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>

using namespace std;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;
using VariantType = std::variant<std::string, int, double>;

// Function to extract the ranking information from the data list
vector<vector<int>> extract_ranking(const vector<map<string, VariantType>> &data_list);

// Function to handle the ties in ranking
// Example input: [[1], [2, 2, 2], [3, 3], [4], [5]]
// Example output: [[1], [2, 2, 2], [5, 5], [7], [8]]
vector<vector<int>> transform_ranking(const vector<vector<int>> &intermediate_ranking);

// Function to update the data list with the new_ranking
vector<map<string, VariantType>> create_new_data_with_rank(const vector<map<string, VariantType>> &data_list, const vector<vector<int>> &transformed_ranking);

// Given the original ranking information, handle the ties by using the helper functions defined
vector<map<string, VariantType>> update_ranking_for_one_target_function(const vector<map<string, VariantType>> &ranking_info);

// Function to convert json to vector of maps
vector<map<string, VariantType>> json_to_vector_map(const ordered_json &j);

// Function to convert vector of maps to json
ordered_json vector_map_to_json(const vector<map<string, VariantType>> &vec);

#endif