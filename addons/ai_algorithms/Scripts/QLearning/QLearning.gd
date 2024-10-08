class_name QLearning

# Observation Spaces are the possible states the agent can be in
# Action Spaces are the possible actions the agent can take
var observation_space: int  # Number of different states the agent can encounter
var action_spaces: int       # Number of different actions the agent can take

# The table that contains the value for each cell in the QLearning algorithm
var QTable: Matrix          # Matrix to store Q-values for state-action pairs

# Modified version of QLearning that enables SARSA (State-Action-Reward-State-Action)
var SARSA: bool = true

# Hyper-parameters
var exploration_probability: float = 1.0 # The probability that the agent will explore instead of exploiting the QTable
var exploration_decreasing_decay: float = 0.01 # Rate at which exploration probability decreases
var min_exploration_probability: float = 0.01 # The minimum value exploration probability can reach
var discounted_factor: float = 0.9 # Discount factor (gamma) for future rewards
var learning_rate: float = 0.2 # Rate at which the agent learns from new information
var decay_per_steps: int = 100  # Number of steps after which exploration probability decays
var steps_completed: int = 0      # Counter for steps taken by the agent

# States
var previous_state: int = -100 # Used to store the previous state for updates
var current_state: int          # Current state of the agent to be updated
var previous_action: int        # Action taken in the previous step, to be used for updates

var done: bool = false           # Flag indicating if the episode is finished
var is_learning: bool = true     # Flag to indicate if the agent is currently learning
var print_debug_info: bool = false # Flag to enable printing debug information

# Constructor function to initialize the QLearning agent
func _init(n_observations: int, n_action_spaces: int, _is_learning: bool = true, not_sarsa: bool = false) -> void:
	observation_space = n_observations  # Set the number of observations
	action_spaces = n_action_spaces      # Set the number of actions
	is_learning = _is_learning            # Set the learning flag
	SARSA = not not_sarsa
	
	QTable = Matrix.new(observation_space, action_spaces)  # Initialize the Q-table as a matrix

# Function to predict the action to take based on the current state and reward
func predict(current_state: int, reward_of_previous_state: float) -> int:
	
	var action_to_take: int  # Variable to hold the action to take
	
	# If a random number is less than exploration probability, explore; otherwise, exploit the QTable
	if randf() < exploration_probability and is_learning:
		action_to_take = randi_range(0, action_spaces - 1)  # Choose a random action
	else:
		action_to_take = QTable.index_of_max_from_row(current_state)  # Choose the best action based on Q-values
	
	if is_learning:
		# Update the Q-table if the agent is in learning mode
		if previous_state != -100:  # Check if there's a valid previous state
			# Update the Q-value for the previous state-action pair
			if not SARSA:
				# Traditional QLearning algo
				QTable.data[previous_state][previous_action] = (1 - learning_rate) * QTable.data[previous_state][previous_action] + \
				learning_rate * (reward_of_previous_state + discounted_factor * QTable.max_from_row(current_state))
			else:
				# SARSA algo
				QTable.data[previous_state][previous_action] = (1 - learning_rate) * QTable.data[previous_state][previous_action] + \
				learning_rate * (reward_of_previous_state + discounted_factor * QTable.data[current_state][action_to_take])
			
		previous_state = current_state  # Update previous state
		previous_action = action_to_take  # Update previous action
		
		# Decay exploration probability after a certain number of steps
		if decay_per_steps != 0:
			if steps_completed != 0 and steps_completed % decay_per_steps == 0:
				exploration_probability = maxf(min_exploration_probability, exploration_probability - exploration_decreasing_decay)  # Decrease exploration probability, but not below the minimum
	if decay_per_steps != 0:
		# Print debug information if enabled and if it's time to do so
		if print_debug_info and steps_completed % decay_per_steps == 0:
			print("Total steps completed:", steps_completed)  # Print total steps taken
			print("Current exploration probability:", exploration_probability)  # Print current exploration probability
			print("Q-Table data:", QTable.data)  # Print Q-table data
			print("-----------------------------------------------------------------------------------------")
	
	steps_completed += 1  # Increment the steps completed counter
	return action_to_take  # Return the action to be taken
