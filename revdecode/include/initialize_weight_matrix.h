// initialize_weight_matrix.h
#ifndef INITIALIZE_WEIGHT_MATRIX_H
#define INITIALIZE_WEIGHT_MATRIX_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace std;
using json = nlohmann::json;

/**
 * @brief Store candidate functions' attributes read from input.
 */
struct CandidateFunction {
    string library_name;
    string library_version;
    string compiler_option;
    string function_name;
    string compiler_unit;
    double similarity_score;
    double emission_probability;
    double uniqueness_score;
    int rank;
};

/**
 * @brief Store the hashed candidate function's attributes.
 */
struct HashedCandidateFunction {
    unsigned int library_name_hash;
    unsigned int library_version_hash;
    unsigned int compiler_option_hash;
    unsigned int function_name_hash;
    unsigned int compiler_unit_hash;
    double similarity_score;
    double emission_probability;
    double uniqueness_score;
    int rank;
};

/**
 * @brief FNV-1a hash helper function.
 *
 * @param str: Original string.
 */
unsigned int fnv1a_hash(const char *str);

/**
 * @brief Use FNV-1a hash to convert candidate functions' string attributes to integers for efficient comparisons.
 *
 * @param original: A vector of 'CandidateFunction' structures containing the original candidate function information.
 * @param hashed: A vector of 'HashedCandidateFunction' structures to be filled with the hashed values of the original candidate functions.
 */
void hash_candidates(const vector<CandidateFunction> &original, vector<HashedCandidateFunction> &hashed);

/**
 * @brief Update 'Unknown' functions' initial similarity score based on the rank-1 candidate function.
 *
 * @param unkown_state_information: A vector of 'CandidateFunction' structs representing 'Unknown' functions.
 * @param rank_1_inforamtion: A reference to a 'CandidateFunction' struct representing the rank-1 functions.
 * @param unknown_promotion_factor: Bonus value factor applied on 'Unknown' functions.
 */
void update_unkown_states_initial_similarity_score(vector<CandidateFunction> &unknown_state_information,
                                                   const CandidateFunction &rank_1_information,
                                                   double unknown_promotion_factor);

/**
 * @brief Initialize 'Unknown' functions' emission probabilities.
 *
 * @param unkown_state_information: A vector of 'CandidateFunction' structs representing 'Unknown' functions.
 * @param ranking : Rank of all candidates belonged to the same target function.
 */
void update_unknown_states_initial_emission_probability(vector<CandidateFunction> &unknown_state_information,
                                                        const vector<CandidateFunction> &ranking);

/**
 * @brief Initialize the state machine, where each state represents the group of attributes belonged to a candidate function, and the likelihood matrix.
 *
 * @param input_file_contents: A list of 'json' objects where each one contains all candidates belonged to a target function.
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
    int &num_transitions);

#endif