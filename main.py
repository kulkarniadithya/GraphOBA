import argparse
import numpy as np
from graph import load_dataset
from state import State
from worker import simulate_worker_label
from belief_propagation import belief_propagation
from policy import policy_exp, policy_opt
from utils import set_seed, compute_accuracy
from config import SAMPLE_SIZE, SEED


class GraphOBAEnv:
    def __init__(self, G, true_labels, split):
        self.G = G
        self.true_labels = true_labels
        self.split = split
        self.state = State(G.nodes())

    def run(self, policy_name, budget):
        train_nodes = list(self.split['train'])
        for t in range(budget):
            candidates = np.random.choice(train_nodes, size=min(SAMPLE_SIZE, len(train_nodes)), replace=False)
            if policy_name == 'exp':
                v = policy_exp(self, candidates, self.true_labels)
            elif policy_name == 'opt':
                v = policy_opt(self, candidates, self.true_labels)
            else:
                raise ValueError("Invalid policy.")
            y = simulate_worker_label(self.true_labels[v])
            self.state.update_posterior(v, y)
            belief_propagation(self.G, self.state, self.true_labels)

        predictions = {v: 1 if p >= 0.5 else -1 for v, p in self.state.marginals.items()}
        test_nodes = self.split['test']
        test_labels = {v: self.true_labels[v] for v in test_nodes}
        test_preds = {v: predictions[v] for v in test_nodes}
        return compute_accuracy(test_preds, test_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed', 'webkb'], required=True)
    parser.add_argument('--policy', choices=['exp', 'opt'], required=True)
    parser.add_argument('--budget', type=int, required=True)
    args = parser.parse_args()

    set_seed(SEED)
    G, true_labels, split = load_dataset(args.dataset)
    env = GraphOBAEnv(G, true_labels, split)
    acc = env.run(args.policy, args.budget)
    print(f"Final test set accuracy on {args.dataset} using {args.policy}: {acc:.4f}")


if __name__ == '__main__':
    main()