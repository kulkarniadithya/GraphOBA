from config import ALPHA, BETA

class State:
    def __init__(self, nodes):
        self.posterior = {v: [ALPHA, BETA] for v in nodes}
        self.marginals = {v: 0.5 for v in nodes}

    def update_posterior(self, v, label):
        if label == 1:
            self.posterior[v][0] += 1
        else:
            self.posterior[v][1] += 1