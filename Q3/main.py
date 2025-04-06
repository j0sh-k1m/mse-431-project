import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threshold_mc import generate_markov_chain
from calc_cost_3pl import total_cost_3pl
from calc_cost_truck import total_cost_truck

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

def get_threshold_and_states(markov_chain, step=50):
    states = [i for i in range(0, markov_chain.shape[0] * step, step)]
    threshold = markov_chain.shape[0] * step
    return threshold, states


if __name__ == '__main__':

    # Generate Constants
    L_BOUND = 100
    U_BOUND = 1800
    STEP = 50

    # 3PL Cost Constants (researched)
    to_feet_cubed = 35.3147
    VARIABLE_COST_FACTOR = 4.98 * to_feet_cubed
    FIXED_COST_3PL = 800
    HOLDING_COST_FACTOR = 0.54 * to_feet_cubed


    # Truck Costs
    tc_900 = 578.6
    tc_1800 = 727

    ### Generate MCs for all possible thresholds [0, 50, 100, ..., 1800]
    # generate_mcs_all_thresholds(u_bound=U_BOUND, l_bound=L_BOUND)

    ## Get the markov chains from CSV files
    markov_chains = []

    for i in range(L_BOUND, U_BOUND + STEP, STEP):
        mat = read_csv_to_2d_numpy_array(f'markov-chains/threshold2-{i}-mc.csv')
        markov_chains.append(mat)

    ## Calculate Long Run probabilities for each Markov Chain
    LONG_RUN_PROBS = {i:[0]*mc.shape[0] for i, mc in enumerate(markov_chains)}

    for i, mc in enumerate(markov_chains):
        lrp = calculate_long_run_probs(mc)
        LONG_RUN_PROBS[i * STEP + 100] = lrp

    total_costs_3pl = total_cost_3pl(
        markov_chains=markov_chains,
        vc_factor=VARIABLE_COST_FACTOR,
        fixed_cost=FIXED_COST_3PL,
        hc_factor=HOLDING_COST_FACTOR
    )

    total_costs_trucks = total_cost_truck(
        markov_chains=markov_chains,
        tc_900=tc_900,
        tc_1800=tc_1800,
        hc_factor=HOLDING_COST_FACTOR,
    )

    # print("3PL", total_costs_3pl, "\n\n\n\n", "Truck", total_costs_trucks)

    # GRAPH
    x_3pl = list(total_costs_3pl.keys())
    y_3pl = list(total_costs_3pl.values())

    x_truck = list(total_costs_trucks.keys())
    y_truck = list(total_costs_trucks.values())

    ### EXTRA GRAPHS ###
    HCF = 0.0
    HOLDING_COST_FACTOR = HCF * to_feet_cubed
    total_costs_3pl_extra = total_cost_3pl(
        markov_chains=markov_chains,
        vc_factor=VARIABLE_COST_FACTOR,
        fixed_cost=FIXED_COST_3PL,
        hc_factor=HOLDING_COST_FACTOR
    )

    x_3pl_extra = list(total_costs_3pl_extra.keys())
    y_3pl_extra = list(total_costs_3pl_extra.values())

    total_costs_trucks_extra = total_cost_truck(
        markov_chains=markov_chains,
        tc_900=tc_900,
        tc_1800=tc_1800,
        hc_factor=HOLDING_COST_FACTOR,
    )

    x_truck_extra = list(total_costs_trucks_extra.keys())
    y_truck_extra = list(total_costs_trucks_extra.values())

    plt.figure(figsize=(10, 6))

    # Plot 3PL costs
    plt.plot(x_3pl, y_3pl, marker='o', linestyle='-', color='blue', label='3PL Costs (HCF=0.54)')

    # Plot Truck costs
    # plt.plot(x_truck, y_truck, marker='s', linestyle='--', color='red', label='Truck Costs (HCF=0.54)')

    # Plot 3PL Extra
    # plt.plot(x_3pl_extra, y_3pl_extra, marker='^', linestyle='-', color='red', label=f'3PL Costs (HCF={HCF})')

    # Plot Truck Extra
    # plt.plot(x_truck_extra, y_truck_extra, marker='s', linestyle='--', color='red', label=f'Truck Costs (HCF={HCF})')

    # Adding labels and title
    plt.xlabel('Threshold')
    plt.ylabel('Avg Cost (per day)')
    plt.title('Avg Costs: 3PL')

    # Add a legend to differentiate the plots
    plt.legend()

    # Optional: add grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()

    # Q4
    # markov_c = read_csv_to_2d_numpy_array(f'markov-chains/final_transition_matrix_correct.csv')
    # print(markov_c.shape)
    #
    # q4_cost_3pl = total_cost_3pl(
    #     markov_chains=[markov_c],
    #     vc_factor=VARIABLE_COST_FACTOR,
    #     fixed_cost=FIXED_COST_3PL,
    #     hc_factor=HOLDING_COST_FACTOR,
    # )
    #
    # q4_cost_trucks = total_cost_truck(
    #     markov_chains=[markov_c],
    #     tc_900=tc_900,
    #     tc_1800=tc_1800,
    #     hc_factor=HOLDING_COST_FACTOR,
    # )
    #
    # print(q4_cost_3pl, "\n\n\n")
    # print(q4_cost_trucks)

