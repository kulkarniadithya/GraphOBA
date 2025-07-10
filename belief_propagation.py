from config import BP_MAX_ITERS


def belief_propagation(G, state, true_labels):
    for _ in range(BP_MAX_ITERS):
        new_marginals = {}
        for v in G.nodes():
            msg_product = 1
            for u in G.neighbors(v):
                msg = state.marginals[u] if true_labels[u] == true_labels[v] else 1 - state.marginals[u]
                msg_product *= msg
            a, b = state.posterior[v]
            prior = a / (a + b)
            new_marginals[v] = (prior + msg_product) / 2
        state.marginals = new_marginals
