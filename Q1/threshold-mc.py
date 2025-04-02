import math
import numpy as np
from itertools import product
import pandas as pd

### INPUT PARAMETERS
LAMBDA_POISSON = 2
MAX_ORDER_QUANTITY = 3

COMPONENTS = {50: 0.3, 100: 0.5, 150: 0.2}
VOLUMES = list(COMPONENTS.keys())
NUM_COMPONENTS = len(VOLUMES)

MAX_VOLUME = 1800
STEP = 50
STATES = np.arange(0, MAX_VOLUME + STEP, STEP)
NUM_STATES = len(STATES)

print("NUMBER OF STATES: ", NUM_STATES)

### CODE/LOGIC STARTS HERE...

# Poisson Probabilities from 0 to 2 inclusive
POISSON_PROBS = [math.exp(-LAMBDA_POISSON) * (LAMBDA_POISSON ** k) / math.factorial(k) for k in range(NUM_COMPONENTS)]

# Poisson Probability for the last component
poisson_end = 1 - sum(POISSON_PROBS)
POISSON_PROBS.append(poisson_end)

def volume_distribution(n):
    """
    Computes the distribution of volumes for n orders using the input parameters stated above
    :param n: number of orders received
    :return: distribution of volumes
    """
    outcomes = {}
    if n == 0:
        outcomes[0] = 1.0

    else:
        for outcome in product(VOLUMES, repeat=n):
            # Get the total volume
            total_volume = sum(outcome)
            prob = np.prod([COMPONENTS[v] for v in outcome])
            outcomes[total_volume] = outcomes.get(total_volume, 0) + prob

    return outcomes

# Get order distribution
ORDER_DISTRIBUTION = [volume_distribution(k) for k in range(MAX_ORDER_QUANTITY+1)]

PROB_DIST = {}
for i in range(MAX_ORDER_QUANTITY+1):

    for vol, prob in ORDER_DISTRIBUTION[i].items():
        PROB_DIST[vol] = PROB_DIST.get(vol, 0) + POISSON_PROBS[i] * prob

# Build the transition matrix
P = np.zeros((NUM_STATES, NUM_STATES))
for i, state in enumerate(STATES):
    for vol, prob in PROB_DIST.items():
        new_volume = state + vol

        if state == MAX_VOLUME and vol == 0:
            j = np.searchsorted(STATES, 0)
        # Otherwise, if new volume exceeds max capacity, carry over the excess
        elif new_volume > MAX_VOLUME:
            carry_over = new_volume - MAX_VOLUME
            j = np.searchsorted(STATES, carry_over)
        else:
            j = np.searchsorted(STATES, new_volume)
        P[i, j] += prob


row_sums = np.sum(P, axis=1)
tolerance = 1e-8
for i, s in enumerate(row_sums):
    if not np.isclose(s, 1.0, atol=tolerance):
        raise ValueError(f"Row {i} should have sum of 1 but got: {s}")

print("Row sums are all equal to 1!")

# Normalize rows to ensure that they sum to 1 (handling floating-point precision)
P = P / P.sum(axis=1, keepdims=True)

df = pd.DataFrame(P, index=STATES, columns=STATES)

# Name of CSV FILE to output to
FILE_NAME = f"threshold-{MAX_VOLUME}-mc.csv"
df.to_csv(f'markov-chains/{FILE_NAME}')
print(f"Saved as markov-chains/{FILE_NAME}")


