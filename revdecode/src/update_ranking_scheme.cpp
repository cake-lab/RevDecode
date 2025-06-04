// update_ranking_scheme.cpp
#include "update_ranking_scheme.h"

using namespace std;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;
using VariantType = std::variant<std::string, int, double>;

// Function to extract the ranking information from the data list
vector<vector<int>> extract_ranking(const vector<map<string, VariantType>> &data_list) {
    // Create a flat list of all ranks
    vector<int> rank_list;
    for (const auto &item : data_list) {
        rank_list.push_back(get<int>(item.at("rank")));
    }

    // Grouping consecutive equal ranks into sublists
    vector<vector<int>> intermediate_ranking;
    while (!rank_list.empty()) {
        int value = rank_list.front();
        rank_list.erase(rank_list.begin());
        vector<int> sublist = {value};
        while (!rank_list.empty() && rank_list.front() == value) {
            sublist.push_back(rank_list.front());
            rank_list.erase(rank_list.begin());
        }
        intermediate_ranking.push_back(sublist);
    }

    return intermediate_ranking;
}

// Function to handle the ties in ranking
// Example input: [[1], [2, 2, 2], [3, 3], [4], [5]]
// Example output: [[1], [2, 2, 2], [5, 5], [7], [8]]
vector<vector<int>> transform_ranking(const vector<vector<int>> &intermediate_ranking) {
    vector<vector<int>> new_ranking;
    int prev_rank = 0;
    for (const auto &sublist : intermediate_ranking) {
        int new_rank = prev_rank + 1;
        new_ranking.push_back(vector<int>(sublist.size(), new_rank));
        prev_rank = new_rank + static_cast<int>(sublist.size()) - 1;
    }

    return new_ranking;
}

// Function to update the data list with the new_ranking
vector<map<string, VariantType>> create_new_data_with_rank(const vector<map<string, VariantType>> &data_list, const vector<vector<int>> &transformed_ranking) {
    vector<map<string, VariantType>> new_data;
    size_t data_index = 0;
    for (const auto &sublist : transformed_ranking) {
        for (int rank : sublist) {
            map<string, VariantType> updated_item = data_list[data_index];
            updated_item["rank"] = rank;
            new_data.push_back(updated_item);
            data_index++;
        }
    }
    return new_data;
}

// Given the original ranking information, handle the ties by using the helper functions defined
vector<map<string, VariantType>> update_ranking_for_one_target_function(const vector<map<string, VariantType>> &ranking_info) {
    auto intermediate = extract_ranking(ranking_info);
    auto transformed = transform_ranking(intermediate);
    auto updated_ranking = create_new_data_with_rank(ranking_info, transformed);
    return updated_ranking;
}

// Function to convert json to vector of maps
vector<map<string, VariantType>> json_to_vector_map(const ordered_json &j) {
    vector<map<string, VariantType>> vec;
    for (const auto &item : j) {
        map<string, VariantType> map;
        for (const auto &element : item.items()) {
            if (element.value().is_string()) {
                map[element.key()] = element.value().get<string>();
            } else if (element.value().is_number_integer()) {
                map[element.key()] = element.value().get<int>();
            } else if (element.value().is_number_float()) {
                double original_value = element.value().get<double>();
                map[element.key()] = original_value;
            }
        }
        vec.push_back(map);
    }
    return vec;
}

// Function to convert vector of maps to json
ordered_json vector_map_to_json(const vector<map<string, VariantType>> &vec) {
    ordered_json j = ordered_json::array();

    for (const auto &map : vec) {
        ordered_json map_json;

        // Add attributes in the specific order
        if (map.find("library_name") != map.end()) {
            if (holds_alternative<string>(map.at("library_name"))) {
                map_json["library_name"] = get<string>(map.at("library_name"));
            }
        }
        if (map.find("library_version") != map.end()) {
            if (holds_alternative<string>(map.at("library_version"))) {
                map_json["library_version"] = get<string>(map.at("library_version"));
            }
        }
        if (map.find("compiler_option") != map.end()) {
            if (holds_alternative<string>(map.at("compiler_option"))) {
                map_json["compiler_option"] = get<string>(map.at("compiler_option"));
            }
        }
        if (map.find("function_name") != map.end()) {
            if (holds_alternative<string>(map.at("function_name"))) {
                map_json["function_name"] = get<string>(map.at("function_name"));
            }
        }
        // Add compiler_unit
        if (map.find("compiler_unit") != map.end()) {
            if (holds_alternative<string>(map.at("compiler_unit"))) {
                map_json["compiler_unit"] = get<string>(map.at("compiler_unit"));
            }
        }
        if (map.find("similarity_score") != map.end()) {
            if (holds_alternative<double>(map.at("similarity_score"))) {
                double value = get<double>(map.at("similarity_score"));
                map_json["similarity_score"] = value;
            }
        }
        // Add emission_probability
        if (map.find("confidence_factor") != map.end()) {
            if (holds_alternative<double>(map.at("confidence_factor"))) {
                double value = get<double>(map.at("confidence_factor"));
                map_json["confidence_factor"] = value;
            }
        }
        // Add uniqueness_score
        if (map.find("uniqueness_score") != map.end()) {
            if (holds_alternative<double>(map.at("uniqueness_score"))) {
                double value = get<double>(map.at("uniqueness_score"));
                map_json["uniqueness_score"] = value;
            }
        }
        if (map.find("rank") != map.end()) {
            if (holds_alternative<int>(map.at("rank"))) {
                map_json["rank"] = get<int>(map.at("rank"));
            }
        }
        if (map.find("original_rank") != map.end()) {
            if (holds_alternative<int>(map.at("original_rank"))) {
                map_json["original_rank"] = get<int>(map.at("original_rank"));
            }
        }

        j.push_back(map_json);
    }

    return j;
}