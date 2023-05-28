import random

def secretary_problem(num_candidates):
    """
    Simulates the Secretary Problem and returns the probability of selecting the best candidate.
    Args:
        num_candidates (int): The number of candidates available.
    Returns:
        float: The probability of selecting the best candidate.
    """
    num_simulations = 100000  # Number of simulations to run
    num_success = 0  # Counter for the number of successful selections
    
    for _ in range(num_simulations):
        candidates = list(range(1, num_candidates + 1))
        best_candidate = max(candidates)
        k = int(num_candidates / 2)  # Threshold value
        
        selected_candidates = random.sample(candidates, k)
        selected_best = max(selected_candidates)
        
        if selected_best == best_candidate:
            num_success += 1
    
    probability_success = num_success / num_simulations
    return probability_success

# Example usage
num_candidates = 10

probability_success = secretary_problem(num_candidates)
print(f"The probability of selecting the best candidate is: {probability_success}")
