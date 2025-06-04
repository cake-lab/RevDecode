// initialize_weight_matrix.cpp
#include "initialize_weight_matrix.h"
#include "utils.h"
#include <cfloat>

using namespace std;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

#define TOP_K 511 // Number of candidates to consider for each target function

/**
 * @brief FNV-1a hash helper function.
 *
 * @param str: Original string.
 */
unsigned int fnv1a_hash(const char *str) {
    // This is a prime number used in the multiplication step. Prime numbers are chosen because they help in spreading
    // the bits more uniformly, reducing the likelihood of hash collisions.
    const unsigned int FNV_prime = 16777619U;

    // This is the initial value of the hash. It's a large, well-distributed number chosen to start the hashing process.
    const unsigned int FNV_offset_basis = 2166136261U;

    unsigned int hash = FNV_offset_basis;

    while (*str) {
        hash ^= static_cast<unsigned char>(*str++);
        hash *= FNV_prime;
    }
    return hash;
}

/**
 * @brief Use FNV-1a hash to convert candidate functions' string attributes to integers for efficient comparisons.
 *
 * @param original: A vector of 'CandidateFunction' structures containing the original candidate function information.
 * @param hashed: A vector of 'HashedCandidateFunction' structures to be filled with the hashed values of the original candidate functions.
 */
void hash_candidates(const vector<CandidateFunction> &original, vector<HashedCandidateFunction> &hashed) {
    hashed.resize(original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        hashed[i].library_name_hash = fnv1a_hash(original[i].library_name.c_str());
        hashed[i].library_version_hash = fnv1a_hash(original[i].library_version.c_str());
        hashed[i].compiler_option_hash = fnv1a_hash(original[i].compiler_option.c_str());
        hashed[i].function_name_hash = fnv1a_hash(original[i].function_name.c_str());
        hashed[i].compiler_unit_hash = fnv1a_hash(original[i].compiler_unit.c_str());
        hashed[i].similarity_score = original[i].similarity_score;
        hashed[i].emission_probability = original[i].emission_probability;
        hashed[i].uniqueness_score = original[i].uniqueness_score;
        hashed[i].rank = original[i].rank;
    }
}

/**
 * @brief Update 'Unknown' functions' initial similarity score based on the rank-1 candidate function.
 *
 * @param unkown_state_information: A vector of 'CandidateFunction' structs representing 'Unknown' functions.
 * @param rank_1_inforamtion: A reference to a 'CandidateFunction' struct representing the rank-1 functions.
 * @param unknown_promotion_factor: Bonus value factor applied on 'Unknown' functions.
 */
void update_unkown_states_initial_similarity_score(vector<CandidateFunction> &unknown_state_information,
                                                   const CandidateFunction &rank_1_information,
                                                   double unknown_promotion_factor) {
    double updated_similarity_score = rank_1_information.similarity_score * unknown_promotion_factor;

    // Ensure the updated similarity score not exceed 1.0
    if (updated_similarity_score > 1.0) {
        updated_similarity_score = 1.0;
    }

    for (auto &item : unknown_state_information) {
        item.similarity_score = updated_similarity_score;
    }
}

/**
 * @brief Initialize 'Unknown' functions' emission probabilities.
 *
 * @param unkown_state_information: A vector of 'CandidateFunction' structs representing 'Unknown' functions.
 * @param ranking : Rank of all candidates belonged to the same target function.
 */
void update_unknown_states_initial_emission_probability(vector<CandidateFunction> &unknown_state_information,
                                                        double unknown_confidence_factor) {
    for (auto &item : unknown_state_information) {
        item.emission_probability = unknown_confidence_factor;
    }
}

/**
 * @brief Initialize the state machine, where each state represents the group of attributes belonged to a candidate function, and the likelihood matrix.
 *
 * @param input_file_contents: A	list of 'json' objects where each one contains all candidates belonged to a target function.
 * @param transition_liklihoods: The initial likelihood matrix. Each entry represents the transition likelihood from a source candidate
 *								to a destination candidate, and the initial value is set as the destination candidate's 'similarity_score'.
 * @param candidate_function_to_state_machine_map: The state machine where each state represents the group of attributes belonged to a candidate function.
 * @param unknown_emission_probability: Initial emission probability for "Unknown" functions.
 * @param unknown_uniqueness_score: Initial unique score for "Unknown" functions.
 * @param unknown_promotion_factor: Variable for boosting the initial similarity score for "Unknown" functions.
 * @param num_candidates_per_function: Record the number of candidates, plus the number of "Unknown"s for each target function.
 * @param num_targets: Record the number of target functions.
 * @param num_transitions: Record the number of transitions, which is equivalent to the size of the likelihood matrix.
 */
void initialize_machine_for_calculate_probabilities(
    const vector<json> &input_file_contents,
    vector<double> &transition_likelihoods,
    vector<CandidateFunction> &candidate_function_to_state_machine_map,
    double unknown_emission_probability,
    double unknown_uniqueness_score,
    double unknown_promotion_factor,
    vector<int> &num_candidates_per_function,
    int &num_targets,
    int &num_transitions) {
    // Create a template for Start and End states
    CandidateFunction start_state_information = {"Start", "Start", "Start", "Start", "Start", 0.0f, 0.0f, 0.0f, -1};
    CandidateFunction finish_state_information = {"Finish", "Finish", "Finish", "Finish", "Finish", 0.0f, 0.0f, 0.0f, -1};

    // Create a template for 'Unknown' functions. Note there might be multiple "Unknown" functions so we store them in a vector
    vector<CandidateFunction> unknown_state_information = {{"Unknown", "Unknown", "Unknown", "Unknown", "Unknown", 0.0f, unknown_emission_probability, unknown_uniqueness_score, 1}};

    // Start to construct the state machine.
    candidate_function_to_state_machine_map.push_back(start_state_information); // Add the start state which transits to the candidates in the first target function

    for (const auto &target_function : input_file_contents) {
        double unknown_confidence_factor = roundToDigits(target_function["unknown_confidence_factor"].get<double>(), 8);
        auto ranking = target_function["ranking"]; // Access the group of candidates in the target function

        vector<json> resized_ranking(ranking.begin(), ranking.begin() + min(ranking.size(), static_cast<size_t>(TOP_K)));

        // Extract the rank-1 candidate for updating 'emission_probability' and 'uniqueness_score' of the 'Unknown' function
        CandidateFunction rank_1_information;
        if (!resized_ranking.empty()) {
            const auto &top_candidate_json = resized_ranking[0];
            rank_1_information.library_name = top_candidate_json["library_name"].get<string>();
            rank_1_information.library_version = top_candidate_json["library_version"].get<string>();
            rank_1_information.compiler_option = top_candidate_json["compiler_option"].get<string>();
            rank_1_information.function_name = top_candidate_json["function_name"].get<string>();
            rank_1_information.compiler_unit = top_candidate_json["compiler_unit"].get<string>();
            rank_1_information.similarity_score = roundToDigits(top_candidate_json["similarity_score"].get<double>(), 8);
            rank_1_information.emission_probability = roundToDigits(top_candidate_json["confidence_factor"].get<double>(), 8);
            rank_1_information.uniqueness_score = roundToDigits(top_candidate_json["uniqueness_score"].get<double>(), 8);
            rank_1_information.rank = top_candidate_json["rank"].get<int>();
        }

        // Insert the candidates of the target function into the state machine
        vector<CandidateFunction> ranking_candidates;
        for (const auto &candidate_json : resized_ranking) {
            CandidateFunction candidate;
            candidate.library_name = candidate_json["library_name"].get<string>();
            candidate.library_version = candidate_json["library_version"].get<string>();
            candidate.compiler_option = candidate_json["compiler_option"].get<string>();
            candidate.function_name = candidate_json["function_name"].get<string>();
            candidate.compiler_unit = candidate_json["compiler_unit"].get<string>();
            candidate.similarity_score = roundToDigits(candidate_json["similarity_score"].get<double>(), 8);
            candidate.emission_probability = roundToDigits(candidate_json["confidence_factor"].get<double>(), 8);
            candidate.uniqueness_score = roundToDigits(candidate_json["uniqueness_score"].get<double>(), 8);
            candidate.rank = candidate_json["rank"].get<int>();
            candidate_function_to_state_machine_map.push_back(candidate);
            ranking_candidates.push_back(candidate);
        }

        // Update the "Unknown" function
        auto updated_unknown_state_information = unknown_state_information;
        update_unkown_states_initial_similarity_score(updated_unknown_state_information, rank_1_information, unknown_promotion_factor);
        update_unknown_states_initial_emission_probability(updated_unknown_state_information, unknown_confidence_factor);

        candidate_function_to_state_machine_map.insert(candidate_function_to_state_machine_map.end(), updated_unknown_state_information.begin(), updated_unknown_state_information.end());

        // Record the number of candidates for this target function (including "Unknown" functions)
        num_candidates_per_function.push_back(static_cast<int>(resized_ranking.size() + updated_unknown_state_information.size()));
    }

    candidate_function_to_state_machine_map.push_back(finish_state_information); // Add the finish state

    // Initialize the 1-D likelihood matrix

    // Calculate the total number of transitions, which is equivalent to the size of the likelihood matrix
    num_targets = static_cast<int>(input_file_contents.size());

    int total_transitions = 0;

    total_transitions += num_candidates_per_function[0]; // Start state to first target function

    for (int t = 1; t < num_targets; ++t) {
        total_transitions += num_candidates_per_function[t - 1] * num_candidates_per_function[t];
    }

    total_transitions += num_candidates_per_function[num_targets - 1];

    num_transitions = total_transitions;
    transition_likelihoods.resize(num_transitions);

    int pos = 0; // Position in the 1-D likelihood matrix

    // Start transitions (from start to first target function candidates)
    int start_candidate_index = 1; // Index of the first candidate after the start state
    int num_first_candidates = num_candidates_per_function[0];

    for (int c = 0; c < num_first_candidates; ++c) {
        double similarity_score = candidate_function_to_state_machine_map[start_candidate_index + c].similarity_score;
        transition_likelihoods[pos++] = similarity_score;
    }

    // Transitions between target functions
    int candidate_offset = 1; // Start from index 1 (after start state)
    for (int t = 1; t < num_targets; ++t) {
        int num_prev_candidates = num_candidates_per_function[t - 1];
        int num_current_candidates = num_candidates_per_function[t];

        // For each destination candidate in target t
        for (int c_i = 0; c_i < num_current_candidates; ++c_i) {
            int current_candidate_index = candidate_offset + num_prev_candidates + c_i;

            // For each source candidate in target t - 1
            for (int s_k = 0; s_k < num_prev_candidates; ++s_k) {
                // Retrieve the transition likelihood
                double transition_likelihood = candidate_function_to_state_machine_map[current_candidate_index].similarity_score;

                // Store the transition likelihood
                transition_likelihoods[pos++] = transition_likelihood;
            }
        }

        candidate_offset += num_prev_candidates;
    }

    // End transitions (from last target function candidates to end state)
    int num_last_candidates = num_candidates_per_function[num_targets - 1];

    for (int c = 0; c < num_last_candidates; ++c) {
        // Assuming a likelihood of 1.0 for transitions to the end state
        transition_likelihoods[pos++] = 1.0;
    }

    // Ensure we've filled the entire array
    assert(pos == num_transitions);
}
