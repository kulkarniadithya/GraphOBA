import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def compute_accuracy(predictions, ground_truth):
    correct = sum(1 for v in ground_truth if predictions[v] == ground_truth[v])
    return correct / len(ground_truth)