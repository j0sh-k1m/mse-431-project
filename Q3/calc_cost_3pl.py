from threshold_mc import compute_poisson_probabilities, compute_probability_distribution
from helpers import get_threshold_and_states, long_run_probs, expected_cycle_length

### 3PL

def _holding_cost(markov_chain, hc_factor):
    lrp = long_run_probs(markov_chain)
    _, states = get_threshold_and_states(markov_chain)

    if len(lrp) != len(states):
        raise ValueError(f"Long run probabilities: {len(lrp)} mismatch with num_states: {len(states)}")

    hc: float = 0
    for i, state in enumerate(states):
        hc += hc_factor * lrp[i] * state

    return hc

def _variable_cost_3pl(threshold, curr_state, steady_state_p, check=450, step=50) -> float:
    lower = threshold - curr_state
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
    s: float = 0.0
    if lower <= check:
        for i in range(lower, check + step, step):
            p_i = prob_dist[i]
            s += p_i * i

        s += curr_state * steady_state_p

    return s

def total_cost_3pl(markov_chains: list, vc_factor: float, fixed_cost: float, hc_factor: float) -> dict[int, float]:
    """
    Calculates the cost using 3PL for each threshold
    :param markov_chains: Markov Chains to calculate cost
    :param vc_factor: variable cost factor
    :param fixed_cost: fixed cost
    :param hc_factor: holding cost factor
    :return: threshold mapped to its respective cost
    """
    total_costs: dict[int, float] = {}
    for mc in markov_chains:
        threshold, states = get_threshold_and_states(mc)

        lrp = long_run_probs(mc)

        vc: float = 0
        for i, state in enumerate(states):
            vc += _variable_cost_3pl(threshold=threshold, curr_state=state, steady_state_p=lrp[i])

        vc *= vc_factor
        hc = _holding_cost(markov_chain=mc, hc_factor=hc_factor)

        cycle_length = expected_cycle_length(threshold)

        # Total Cost (per day)
        total_costs[threshold] = (fixed_cost/cycle_length) + (vc/cycle_length) + hc

        # print(f"Threshold {threshold};\nVariable Cost (per day): {vc/cycle_length}, Holding Cost (per day): {hc}, Fixed Cost (per day): {fixed_cost/cycle_length}")
        # print(f"Total Cost (Per Day): {total_costs[threshold]}")
        # print(f"Total Cost (Per Cycle): {total_costs[threshold]*cycle_length}" + "\n")

    return total_costs


if __name__ == '__main__':
    print()
