// pre_forward_kernels.cu
#include "pre_forward_kernels.h"

/**
 * @brief This kernel computes the sum of `emission_probability` and `uniqueness_score` for each candidate.
 * @kernel properties:
 *		   number of threads = number of candidate functions
 *		   threads per block = max threads per block on device
 *		   number of blocks = number of candidate functions / max threads per block on device
 *
 * @param d_candidates: Device array of `HashedCandidateFunction` structures.
 * @param d_state_emission_uniqueness: Device array to store the computed `emission_probability` + `uniqueness_score` results for each candidate.
 * @param num_states: The total number of states to process.
 */
__global__ void computeEmissionUniquenessKernel(
    HashedCandidateFunction *d_candidates, double *d_state_emission_uniqueness, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        d_state_emission_uniqueness[idx] = d_candidates[idx].emission_probability + d_candidates[idx].uniqueness_score;
    }
}

/**
* @brief This kernel applies the computed `emission_probability` + `uniqueness_score` additions,
                 and `bonus_factor` based on the contextual information to each transition likelihood.
* @kernel properties:
*		  number of threads = number of transitions
*		  threads per block = max threads per block on device
*		  number of blocks = number of transitions / max threads per block on device
*
* @param transition_likelihoods: 1-D array representing transition likelihoods.
* @param d_candidates: Device array of `HashedCandidateFunction` structures.
* @param d_state_emission_uniqueness: Device array of precomputed `emission_probability` + `uniqueness_score` for each candidate.
* @param num_candidates_per_function: Array representing the number of candidates for each target function.
* @param bonus_factor: The bonus value to apply to the transitions.
* @param unknown_hash: The FNV-1a hash value for "Unknown" to identify 'Unknown' functions.
* @param num_targets: The number of target functions.
* @param num_transitions: The total number of transitions (size of `transition_likelihoods`).
*/
__global__ void preForwardKernel(
    double *transition_likelihoods,
    HashedCandidateFunction *d_candidates,
    double *d_state_emission_uniqueness,
    int *num_candidates_per_function,
    double bonus_factor,
    unsigned int unknown_hash,
    int num_targets,
    int num_transitions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_transitions) {
        return;
    }

    // Compute the indices for source and destination candidates based on thread index
    // We need to map thread index to the corresponding transition

    int pos = idx;            // Position in the `transition_likelihoods` array
    int candidate_offset = 1; // Start from index 1 (after start state)

    // Variables to hold source and destination candidate indices in `candidate_function_to_stat_machine_map`
    int source_candidate_idx = -1;
    int dest_candidate_idx = -1;

    int num_first_candidates = num_candidates_per_function[0];
    if (pos < num_first_candidates) { // Transition from start state to first target function candidates
        source_candidate_idx = 0;
        dest_candidate_idx = candidate_offset + pos;
    } else {
        pos -= num_first_candidates; // Transitions between target functions

        for (int t = 1; t < num_targets; ++t) {
            int num_prev_candidates = num_candidates_per_function[t - 1];
            int num_current_candidates = num_candidates_per_function[t];
            int num_transitions_in_layer = num_prev_candidates * num_current_candidates;

            if (pos < num_transitions_in_layer) {
                int s_k = pos % num_prev_candidates; // Source candidate within previous target function
                int c_i = pos / num_prev_candidates; // Destination candidate within current target function

                source_candidate_idx = candidate_offset + s_k;
                dest_candidate_idx = candidate_offset + num_prev_candidates + c_i;

                break; // Successfully locate the layer
            } else {   // Move to the next layer
                pos -= num_transitions_in_layer;
                candidate_offset += num_prev_candidates;
            }
        }

        // If not found yet, handle the transitions from the last target function to the end state
        if (source_candidate_idx == -1) {
            int num_last_candidates = num_candidates_per_function[num_targets - 1];
            if (pos < num_last_candidates) {
                source_candidate_idx = candidate_offset + pos;               // Candidate in last target function
                dest_candidate_idx = candidate_offset + num_last_candidates; // End state index
            }
        }
    }

    // Ensure indices are valid
    if (source_candidate_idx == -1 || dest_candidate_idx == -1) {
        return; // Invalid indices, exit
    }

    // For each entry in 'transition_likelihood', the value will be updated as
    // 'similarity_score' + 'uniqueness_score' + 'emission_probability' + 'bonus_factor'

    // Apply 'bonus_factor'
    if (d_candidates[source_candidate_idx].library_name_hash != unknown_hash &&
        d_candidates[dest_candidate_idx].library_name_hash != unknown_hash) { // Skip bonus application if either candidate is "Unknown"

        if (d_candidates[source_candidate_idx].library_name_hash == d_candidates[dest_candidate_idx].library_name_hash) {
            if (d_candidates[source_candidate_idx].function_name_hash == d_candidates[dest_candidate_idx].function_name_hash &&
                d_candidates[source_candidate_idx].function_name_hash != unknown_hash) {
                transition_likelihoods[idx] = 0.0f; // Non-transition
            } else {
                transition_likelihoods[idx] += bonus_factor; // Transition within the same library, apply bonus
                if (d_candidates[source_candidate_idx].library_version_hash == d_candidates[dest_candidate_idx].library_version_hash &&
                    d_candidates[source_candidate_idx].library_version_hash != unknown_hash) {
                    transition_likelihoods[idx] += 0.03f; // Same library version, extra bonus

                    if (d_candidates[source_candidate_idx].compiler_option_hash == d_candidates[dest_candidate_idx].compiler_option_hash &&
                        d_candidates[source_candidate_idx].compiler_option_hash != unknown_hash) {
                        transition_likelihoods[idx] += 0.02f; // Same compiler option, extra bonus
                    }
                }
                if (d_candidates[source_candidate_idx].compiler_unit_hash == d_candidates[dest_candidate_idx].compiler_unit_hash &&
                    d_candidates[source_candidate_idx].compiler_unit_hash != unknown_hash) {
                    transition_likelihoods[idx] += 0.05f; // Same compiler unit, extra bonus
                }
            }
        }
    }

    // Apply the precomputed emission_uniqueness product
    transition_likelihoods[idx] += d_state_emission_uniqueness[dest_candidate_idx];
}