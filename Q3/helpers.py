import numpy as np
from threshold_mc import compute_poisson_probabilities, compute_probability_distribution

def get_threshold_and_states(markov_chain, step=50):
    states = [i for i in range(0, markov_chain.shape[0] * step, step)]
    threshold = markov_chain.shape[0] * step
    return threshold, states

def long_run_probs(markov_chain):

    n = markov_chain.shape[0]

    # Construct the equation (P - I)x = 0
    A = markov_chain.T - np.eye(n)

    # Replace one equation with the sum constraint
    A[-1] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1  # Sum constraint

    # Solve the linear system
    steady_state = np.linalg.solve(A, b)

    return steady_state

def get_prob_dist():
    poisson_probs = compute_poisson_probabilities(
        lambda_poisson=2,
        max_order_quantity=3
    )
    prob_dist = compute_probability_distribution(
        max_order_quantity=3,
        components={50: 0.3, 100: 0.5, 150: 0.2},
        volumes=[50, 100, 150],
        poisson_probs=poisson_probs
    )
    return prob_dist

def expected_cycle_length(threshold):
    expected_volume_per_order = 50 * 0.3 + 100 * 0.5 + 150 * 0.2
    poisson_dist = compute_poisson_probabilities(lambda_poisson=2, max_order_quantity=3)
    expected_orders_per_day = sum([i*poisson_dist[i] for i in range(len(poisson_dist))])
    expected_volume_per_day = expected_volume_per_order * expected_orders_per_day

    return threshold / expected_volume_per_day
