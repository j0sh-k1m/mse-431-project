import math
import numpy as np
from itertools import product
import pandas as pd

def compute_poisson_probabilities(lambda_poisson, max_order_quantity):
    """
    Computes Poisson probabilities for orders 0 to max_order_quantity-1,
    and then a tail probability for orders >= max_order_quantity.
    """
    probs = [math.exp(-lambda_poisson) * (lambda_poisson ** k) / math.factorial(k)
             for k in range(max_order_quantity)]
    probs.append(1 - sum(probs))  # Tail probability for orders >= max_order_quantity
    return probs

def volume_distribution(n, components, volumes):
    """
    Computes the distribution of volumes for n orders.
    :param n: number of orders received.
    :param components: dict mapping component volume to probability.
    :param volumes: list of component volumes.
    :return: dictionary mapping total volume to its probability.
    """
    outcomes = {0: 1.0} if n == 0 else {}
    if n > 0:
        for outcome in product(volumes, repeat=n):
            total_volume = sum(outcome)
            prob = np.prod([components[v] for v in outcome])
            outcomes[total_volume] = outcomes.get(total_volume, 0) + prob
    return outcomes

def compute_probability_distribution(max_order_quantity, components, volumes, poisson_probs):
    """
    Computes the overall probability distribution for additional volumes based on the order distribution.
    """
    order_distribution = [volume_distribution(k, components, volumes)
                          for k in range(max_order_quantity + 1)]
    prob_dist = {}
    for i in range(max_order_quantity + 1):
        for vol, prob in order_distribution[i].items():
            prob_dist[vol] = prob_dist.get(vol, 0) + poisson_probs[i] * prob
    return prob_dist

def build_transition_matrix(states, max_volume, prob_dist):
    """
    Builds the transition matrix P where rows correspond to the current state (inventory)
    and columns to the next state after adding production (with carryover if over capacity).
    """
    num_states = len(states)
    P = np.zeros((num_states, num_states))

    for i, state in enumerate(states):
        for vol, prob in prob_dist.items():
            new_volume = state + vol
            # Special rule: if you're at full capacity and get zero additional volume,
            # then you ship out all inventory (go to state 0)
            if state == max_volume and vol == 0:
                j = np.searchsorted(states, 0)
            elif new_volume > max_volume:
                carry_over = new_volume - max_volume
                j = np.searchsorted(states, carry_over)
            else:
                j = np.searchsorted(states, new_volume)
            P[i, j] += prob

    return P / P.sum(axis=1, keepdims=True)  # Normalize rows

def generate_markov_chain(lambda_poisson, max_order_quantity, components, max_volume, step, file_path):
    """
    Generates the Markov transition matrix and saves it as a CSV file.
    :param lambda_poisson: λ for the Poisson process.
    :param max_order_quantity: Maximum number of orders per period.
    :param components: Dictionary mapping component volumes to their probabilities.
    :param max_volume: Maximum inventory capacity (threshold value).
    :param step: Increment step for states.
    :param file_path: File path to save the CSV output.
    :return: Transition matrix P and the states array.
    """
    volumes = list(components.keys())
    states = np.arange(0, max_volume + step, step)

    # Use max_order_quantity for computing Poisson probabilities.
    poisson_probs = compute_poisson_probabilities(lambda_poisson, max_order_quantity)
    prob_dist = compute_probability_distribution(max_order_quantity, components, volumes, poisson_probs)
    P = build_transition_matrix(states, max_volume, prob_dist)

    row_sums = np.sum(P, axis=1)
    tolerance = 1e-8
    for i, s in enumerate(row_sums):
        if not np.isclose(s, 1.0, atol=tolerance):
            raise ValueError(f"Row {i} should have sum of 1 but got: {s}")

    df = pd.DataFrame(P, index=states, columns=states)
    df.to_csv(file_path)
    print(f"Saved as {file_path}")

    return P, states

# Example usage:
THRESHOLD = 1800
if __name__ == "__main__":
    P, states = generate_markov_chain(
        lambda_poisson=2,
        max_order_quantity=3,  # Orders 0, 1, 2 and tail for >=3 orders.
        components={50: 0.3, 100: 0.5, 150: 0.2},
        max_volume=THRESHOLD,
        step=50,
        file_path=f'markov-chains/threshold-{THRESHOLD}-mc.csv'
    )
