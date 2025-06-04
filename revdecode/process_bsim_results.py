import argparse
import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path
from copy import deepcopy

from update_ranking_scheme import updating_ranking_for_one_target_function


class SigmoidNormalizer:
    def __init__(self, data=None, mean=None, std=None):
        # If data is provided, calculate mean and std
        # Otherwise, use the provided mean and std
        if data is not None:
            self.mean = np.mean(data)
            self.std = np.std(data)
        elif mean is not None and std is not None:
            # -1.942160201157363, 26.22625775435414
            self.mean = mean
            self.std = std

    def normalize(self, x):
        centered_x = x - self.mean
        scaled_x = centered_x / self.std
        normalized_x = 1 / (1 + np.exp(-scaled_x))
        return normalized_x


def process_results(file_path, output_dir, corpus_ground_truths, self_confidences, library_scores):
    
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    normalizer = SigmoidNormalizer(mean=-1.942160201157363, std=26.22625775435414)
    
    new_results = []
    for target_function in results:
        contents = {}
        function_name = target_function.split("____")[1].split('.')[0]

        contents["Target_function"] = function_name
        contents["Corpus_version"] = 'Sampled Corpus'
        try:
            contents["self_confidence_score"] = self_confidences[target_function][0]
            contents["self_confidence_factor"] = self_confidences[target_function][1]
            contents["unknown_confidence_score"] = self_confidences[target_function][2]
            contents["unknown_confidence_factor"] = self_confidences[target_function][3]
        except KeyError:
            contents["self_confidence_score"] = 0.0
            contents["self_confidence_factor"] = 0.0
            contents["unknown_confidence_score"] = 0.0
            contents["unknown_confidence_factor"] = 0.0
        ranking = results[target_function]
        contents["ranking"] = []
        sorted_ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1][0], reverse=True)}
        rank = 1
        highest_similarity = 0
        for match in sorted_ranking:
            information = deepcopy(corpus_ground_truths[match])
            similarity_score, confidence_score = sorted_ranking[match]
            information["similarity_score"] = sorted_ranking[match][0]
            information["confidence_score"] = sorted_ranking[match][1]
            information["confidence_factor"] = normalizer.normalize(sorted_ranking[match][1])
            # uniqeuness_score is the library_score.
            information["uniqueness_score"] = library_scores.get(match.split("____")[0])
            if similarity_score > highest_similarity:
                highest_similarity = similarity_score
                information["rank"] = rank
            elif similarity_score == highest_similarity:
                information["rank"] = rank
            else:
                highest_similarity = similarity_score
                rank += 1
                information["rank"] = rank
            contents["ranking"].append(information)
        temp = updating_ranking_for_one_target_function(deepcopy(contents["ranking"]))
        contents["ranking"] = deepcopy(temp)

        new_results.append(deepcopy(contents))
    
    target_library = Path(file_path).stem
    
    output_file = f'{output_dir}/{target_library}.json'
    with open(output_file, 'w') as json_output:
        json.dump(new_results, json_output, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process BSim results for Graph Construction.")
    parser.add_argument("--ranking_dir", help="Path to the file containing raw BSim results.")
    parser.add_argument("--corpus_ground_truth_file", help="Path to the corpus ground truth file.")
    parser.add_argument("--self_confidence_file", help="Path to the self-confidence factors file.")
    parser.add_argument("--library_scores_file", help="Path to the library scores file.")
    parser.add_argument("--output_dir", help="Directory to save the processed results.")

    args = parser.parse_args()


    with open(args.corpus_ground_truth_file, 'r') as f:
        corpus_ground_truths = json.load(f)

    with open(args.self_confidence_file, "r") as f:
        self_confidences = json.load(f)

    with open(args.library_scores_file, "r") as f:
        library_scores = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    
    files = os.listdir(args.ranking_dir)
    for file in files:
        file_path = os.path.join(args.ranking_dir, file)
        process_results(file_path, args.output_dir, corpus_ground_truths, self_confidences, library_scores)
    print("All files processed successfully.")


if __name__ == "__main__":
    main()
