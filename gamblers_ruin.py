import random

def gamblers_ruin(initial_money, goal, win_probability):
    """
    Simulates the Gambler's Ruin problem and returns the probability of losing all money before reaching the goal.
    Args:
        initial_money (int): The initial amount of money the gambler has.
        goal (int): The target amount of money the gambler wants to reach.
        win_probability (float): The probability of winning each round.
    Returns:
        float: The probability of losing all money before reaching the goal.
    """
    num_simulations = 100000  # Number of simulations to run
    num_loss = 0  # Counter for the number of times the gambler loses
    
    for _ in range(num_simulations):
        money = initial_money
        
        while money > 0 and money < goal:
            if random.random() < win_probability:
                money += 1  # Win: increase money by 1
            else:
                money -= 1  # Loss: decrease money by 1
        
        if money == 0:
            num_loss += 1
    
    probability_loss = num_loss / num_simulations
    return probability_loss

# Example usage
initial_money = 10
goal = 20
win_probability = 0.5

probability_loss = gamblers_ruin(initial_money, goal, win_probability)
print(f"The probability of losing all money before reaching the goal is: {probability_loss}")
