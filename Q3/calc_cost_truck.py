from helpers import get_threshold_and_states, expected_cycle_length
from calc_cost_3pl import _holding_cost

# Truck
def total_cost_truck(markov_chains, tc_900, tc_1800, hc_factor):

    total_costs: dict[int, float] = {}

    for j, mc in enumerate(markov_chains):
        threshold, states = get_threshold_and_states(mc)
        cycle_length = expected_cycle_length(threshold)

        total_cost_900: float = 0.0
        total_cost_1800: float = 0.0

        if 100 <= threshold <= 500:
            total_cost_900 += tc_900 / cycle_length

        elif 550 <= threshold <= 1800:
            total_cost_1800 += tc_1800 / cycle_length

        hc = _holding_cost(mc, hc_factor=hc_factor)

        total_costs[threshold] = total_cost_900 + total_cost_1800 + hc

    return total_costs