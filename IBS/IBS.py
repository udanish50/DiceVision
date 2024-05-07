import numpy as np

def get_numerical_score_from_bins(value, bins, scores):
    """
    Converts a value to a numerical score based on the provided bins and corresponding scores.

    :param value: The value to convert.
    :param bins: A list of bin edges, from highest to lowest.
    :param scores: A list of scores corresponding to the bins.
    :return: The numerical score for the value.
    """
    # Check if the value is less than or equal to the smallest bin
    if value <= bins[-1]:
        return scores[-1]
    
    for i in range(len(bins) - 1):
        # Check which bin the value falls into
        if bins[i] <= value < bins[i+1]:
            # Linear interpolation for scoring within the bin range
            score_range = scores[i+1] - scores[i]
            bin_range = bins[i+1] - bins[i]
            return scores[i] + ((value - bins[i]) / bin_range) * score_range
    
    # If the value is greater than the highest bin
    return scores[0]

# Example bins and scores
bins = [float('inf'), 150, 100, 50, 10, 0]  # Bins for some metric thresholds
scores_scale = [1, 2, 3, 4, 5]  # Corresponding scores from lowest to highest

# Example scores to be classified
scores = {
    'Stable Diffusions': 6.398260116577148,
    'DELL2': 3.294837713241577,
    'Glide':  4.792013168334961,
    'DELL3':  8.955540657043457
}

# Calculate numerical scores for each model
numerical_scores = {model: get_numerical_score_from_bins(score, bins, scores_scale)
                    for model, score in scores.items()}

numerical_scores

rescaled_scores = np.array([2.88])
human_scores = np.array([3.98])

#, 4.90, 4.71, 4.20
# , 3.63, 2.04, 2.75

# Calculate MAD
mad = np.mean(np.abs(human_scores - rescaled_scores ))

# Calculate MAPE
mape = np.mean(np.abs((human_scores - rescaled_scores) / human_scores)) * 100

mad, mape
