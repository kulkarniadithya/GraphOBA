import numpy as np

def simulate_worker_label(true_label):
    theta = 0.93 if true_label == 1 else 0.07
    return 1 if np.random.rand() < theta else -1