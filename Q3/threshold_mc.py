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

def build_transition_matrix(states, max_volume, prob_dist, carry_over: bool):
    # Modify states based on carry_over condition:
    # modified_states = states if carry_over else states[:-1]
    modified_states = states[:-1]
    num_states = len(modified_states)
    P = np.zeros((num_states, num_states))

    for i, state in enumerate(modified_states):
        for vol, prob in prob_dist.items():
            new_volume = state + vol

            if carry_over:
                if state == max_volume and vol == 0:
                    j = np.searchsorted(modified_states, 0)
                elif new_volume >= max_volume:
                    extra = new_volume - max_volume
                    j = np.searchsorted(modified_states, extra)
                else:
                    j = np.searchsorted(modified_states, new_volume)
            else:
                if new_volume >= max_volume:
                    j = 0
                else:
                    j = np.searchsorted(modified_states, new_volume)
            P[i, j] += prob

    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)
    return P, modified_states # Normalize rows

def generate_markov_chain(lambda_poisson, max_order_quantity, components, max_volume, step, file_path, carry_over: bool):
    """
    Generates the Markov transition matrix and saves it as a CSV file.
    :param carry_over: Boolean whether to consider carry over probabilities
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

    poisson_probs = compute_poisson_probabilities(lambda_poisson, max_order_quantity)
    prob_dist = compute_probability_distribution(max_order_quantity, components, volumes, poisson_probs)

    P, modified_states = build_transition_matrix(states, max_volume, prob_dist, carry_over=carry_over)

    # Verify that row sums are 1
    row_sums = np.sum(P, axis=1)
    tolerance = 1e-8
    for i, s in enumerate(row_sums):
        if not np.isclose(s, 1.0, atol=tolerance):
            raise ValueError(f"Row {i} should have sum of 1 but got: {s}")

    df = pd.DataFrame(P, index=modified_states, columns=modified_states)
    df.to_csv(file_path)
    print(f"Saved as {file_path}")

    return P, modified_states

# Example usage:
THRESHOLD = 700
if __name__ == "__main__":
    mat, sts = generate_markov_chain(
        lambda_poisson=2,
        max_order_quantity=3,
        components={50: 0.3, 100: 0.5, 150: 0.2},
        max_volume=THRESHOLD,
        step=50,
        file_path=f'markov-chains/threshold2-{THRESHOLD}-mc.csv',
        carry_over=False,
    )