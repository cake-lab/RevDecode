// pre_forward_kernels.h
#ifndef PRE_FORWARD_KERNELS_H
#define PRE_FORWARD_KERNELS_H

#include "device_launch_parameters.h"
#include "initialize_weight_matrix.h"
#include <cmath>

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
    HashedCandidateFunction *d_candidates, double *d_state_emission_uniqueness, int num_states);

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
    int num_transitions);

#endif