from fileinput import filename
from typing import List

import numpy as np
import pandas as pd
from threshold_mc import generate_markov_chain

### Steps/Tasks
# 1. Enumerate over all Possible tresholds 0, 50, ..., 1800 inclusive
# 2. Read in CSV files and convert into NP array
# 3. Calculate the long-run P's (Pi_1, Pi_2, ...)


def read_csv_to_2d_numpy_array(file: str) -> np.ndarray:
    """Reads a CSV file into a 2D NumPy array."""
    try:
        df = pd.read_csv(file, header=None)

        df.drop(index=0, columns=0, inplace=True)

        matrix = df.to_numpy()
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError
    except Exception as e:
        raise Exception(e)

def generate_mcs_all_thresholds(l_bound: int, u_bound: int, step: int = 50):
    for threshold in range(l_bound, u_bound + step, step):

        three_pl = True if threshold >= 1450 else False

        generate_markov_chain(
            lambda_poisson=2,
            max_order_quantity=3,
            components={50: 0.3, 100: 0.5, 150: 0.2},
            max_volume=threshold,
            step=50,
            file_path=f'markov-chains/threshold2-{threshold}-mc.csv',
            carry_over=three_pl # need to figure out to calculate this value
        )

def calculate_long_run_probs(P: np.ndarray):
    n = P.shape[0]

    # Construct the equation (P - I)x = 0
    A = P.T - np.eye(n)

    # Replace one equation with the sum constraint
    A[-1] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1  # Sum constraint

    # Solve the linear system
    steady_state = np.linalg.solve(A, b)

    return steady_state


if __name__ == '__main__':
    L_BOUND = 100
    U_BOUND = 1800
    STEP = 50

    ### Generate MCs for all possible thresholds [0, 50, 100, ..., 1800]
    generate_mcs_all_thresholds(u_bound=U_BOUND, l_bound=L_BOUND)

    ### Get the markov chains from CSV files
    # markov_chains = []
    #
    # for i in range(L_BOUND, U_BOUND + STEP, STEP):
    #     mat = read_csv_to_2d_numpy_array(f'markov-chains/threshold2-{i}-mc.csv')
    #     markov_chains.append(mat)

    ### Calculate Long Run probabilities for each Markov Chain
    # LONG_RUN_PROBS = {i:[0]*mc.shape[0] for i, mc in enumerate(markov_chains)}
    #
    # for i, mc in enumerate(markov_chains):
    #     lrp = calculate_long_run_probs(mc)
    #     LONG_RUN_PROBS[i] = lrp

    ### Calculate Average Cost (per day)
    # Assume that we do not pay the $800 fixed cost per shipment
    # Avg Holding Cost + Avg Shipping Cost + Truck Rental

    '''
        Avg Cost (per day) 
        
        LRP = Long Run Probabilities 
        
        
        Avg Holding Cost (per day) = LRP * state * holding cost
        
        Expected Cycle length (days) =  
        
        Avg Shipping Cost  = (threshold value * shipping cost) / E[cycle length]
    '''