import argparse
import os
import sys
import pickle
import json
import numpy as np


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


def normalize_self_confidence_score(self_confidences, output_file_path):
    
    normalizer = SigmoidNormalizer(mean=-1.942160201157363, std=26.22625775435414)
    print("Normalizer created.")

    # Add confidence factor to all functions
    for function in self_confidences:
        scores = [self_confidences[function]] 
        normalized_score = normalizer.normalize(scores[0])
        unknown_confidence_score = scores[0] * 0.85  # unknown confidence score is 85% of target function's self confidence score
        normalized_unknown_confidence_score = normalizer.normalize(unknown_confidence_score)
        scores.append(normalized_score)
        scores.append(unknown_confidence_score)
        scores.append(normalized_unknown_confidence_score)
        self_confidences[function] = scores
    print("Confidence factors added to all functions.")

    # Save the updated self_confidences with normalized scores
    with open(output_file_path, "w") as f:
        json.dump(self_confidences, f)


def assemble_self_confidence_scores(input_file_path):
    self_confidence_files = os.listdir(input_file_path)

    self_confidences = {}

    for file in self_confidence_files:
        with open(os.path.join(input_file_path, file), "rb") as f:
            results = pickle.load(f)
        for target_function in results:
            function_name = target_function
            ranking = results[target_function]
            for match in ranking:
                if function_name != match:
                    print(f"Function name {function_name} does not match with match {match}.")
                    sys.exit(1)
                self_confidences[match] = ranking[match][1]
    
    return self_confidences


def main():
    parser = argparse.ArgumentParser(description="Normalize self confidence scores.")
    parser.add_argument("--input_file_path", help="Path to the input files containing self confidence scores.")
    parser.add_argument("--output_file_path", help="Path to the output file to save normalized self confidence scores.")
    args = parser.parse_args()

    self_confidences = assemble_self_confidence_scores(args.input_file_path)
    print("Self confidence scores assembled.")

    normalize_self_confidence_score(self_confidences, args.output_file_path)  # Call the normalization function
    print("Self confidence scores normalized.")


if __name__ == "__main__":
    main()
    
