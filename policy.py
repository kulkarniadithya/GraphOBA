import numpy as np
from copy import deepcopy
from worker import simulate_worker_label
from belief_propagation import belief_propagation
from config import ALPHA, BETA

def compute_reward(env, v, assign_label, true_labels):
    temp_env = deepcopy(env)
    temp_env.state.update_posterior(v, assign_label)
    belief_propagation(temp_env.G, temp_env.state, true_labels)
    return sum(max(p, 1 - p) for p in temp_env.state.marginals.values())

def policy_exp(env, candidates, true_labels):
    best_v, best_reward = None, -np.inf
    for v in candidates:
        a, b = env.state.posterior[v]
        p1 = (ALPHA + a) / (ALPHA + a + BETA + b)
        p2 = 1 - p1
        r1 = compute_reward(env, v, 1, true_labels)
        r2 = compute_reward(env, v, -1, true_labels)
        reward = p1 * r1 + p2 * r2
        if reward > best_reward:
            best_v, best_reward = v, reward
    return best_v

def policy_opt(env, candidates, true_labels):
    best_v, best_reward = None, -np.inf
    for v in candidates:
        r1 = compute_reward(env, v, 1, true_labels)
        r2 = compute_reward(env, v, -1, true_labels)
        reward = max(r1, r2)
        if reward > best_reward:
            best_v, best_reward = v, reward
    return best_v