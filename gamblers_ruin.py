
def calculate_probability(starting_money, goal, win_probability, tolerance=1e-10):
    """
    Calculate the probability of reaching the goal starting with starting_money, given the win_probability.
    This function uses a dynamic programming approach to iteratively estimate the probability until it converges.

    Args:
    starting_money (int): Initial amount of money.
    goal (int): Target amount of money.
    win_probability (float): Probability of winning in a single bet.
    tolerance (float, optional): Tolerance for the convergence of probability estimation. Default is 1e-10.

    Returns:
    float: The estimated probability of reaching the goal from the starting_money.
    """
    
    # The number of states equals the goal amount plus one (including zero)
    num_states = goal + 1

    # Initialize the probability array with zeros, and set the probability of reaching the goal to 1
    p = [0 for _ in range(num_states)]
    p[goal] = 1.0

    # Initialize the delta (the maximum change in probabilities in each iteration) to a large value
    delta = 1.0 

    # Iterate until the maximum change in probabilities (delta) is less than the specified tolerance
    while delta > tolerance:
        # Reset delta to zero at the start of each iteration
        delta = 0.0 

        # Iterate over each possible state (amount of money), excluding zero and the goal
        for i in range(1, goal):
            # Remember the current probability for this state
            old_p = p[i]

            # Update the probability for this state based on the probabilities of winning and losing the next bet
            p[i] = win_probability * p[i + 1] + (1 - win_probability) * p[i - 1]

            # Update delta to the maximum change observed in this iteration
            delta = max(delta, abs(old_p - p[i]))

    # Return the estimated probability of reaching the goal from the starting amount of money
    return p[starting_money]


# Set the parameters for the calculation
starting_money = 100
goal = 200
win_probability = 0.49

# Calculate and print the probability
print(calculate_probability(starting_money, goal, win_probability))
