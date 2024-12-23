class_name QLearning

# This class implements Q-Learning and SARSA algorithms for reinforcement learning
# Q-Learning is a model-free algorithm that learns to make optimal decisions by
# updating a Q-table based on rewards received from the environment

# Core state and action space definitions
var observation_space: int # Defines how many different states are possible in the environment
var action_spaces: int # Defines how many different actions the agent can perform
var QTable: Matrix # Stores the learned Q-values for each state-action pair

# SARSA vs Q-Learning switch
# SARSA uses the actual next action, while Q-Learning uses the best possible next action
var SARSA: bool = true

# Hyperparameters that control the learning process:
# - exploration_probability: Controls random vs learned actions (epsilon)
# - exploration_decreasing_decay: How fast to reduce exploration
# - min_exploration_probability: Minimum exploration rate
# - discounted_factor: Importance of future rewards (gamma)
# - learning_rate: Speed of learning new information (alpha)
var exploration_probability: float = 1.0 # Higher value means more exploration
var exploration_decreasing_decay: float = 0.01 # How quickly exploration reduces
var min_exploration_probability: float = 0.01 # Minimum exploration threshold
var discounted_factor: float = 0.9 # How much future rewards matter (gamma)
var learning_rate: float = 0.2 # How quickly new information overrides old (alpha)
var decay_per_steps: int = 100 # Steps before reducing exploration
var steps_completed: int = 0 # Total steps taken by agent

# Track states and actions between steps for temporal difference learning
var previous_state: int = -100 # Special value -100 indicates no previous state
var current_state: int # Current environment state
var previous_action: int # Last action taken by agent

# Control flags
var done: bool = false # Indicates if current episode is complete
var is_learning: bool = true # Enables/disables learning updates
var print_debug_info: bool = false # Controls debug output

# Constructor function to initialize the QLearning agent
func _init(n_observations: int, n_action_spaces: int, _is_learning: bool = true, not_sarsa: bool = false) -> void:
	observation_space = n_observations # Set the number of observations
	action_spaces = n_action_spaces # Set the number of actions
	is_learning = _is_learning # Set the learning flag
	SARSA = not not_sarsa
	
	QTable = Matrix.new(observation_space, action_spaces) # Initialize the Q-table as a matrix

# Main prediction function that:
# 1. Chooses between exploration and exploitation
# 2. Updates Q-values based on rewards
# 3. Handles exploration decay
func predict(current_state: int, reward_of_previous_state: float) -> int:
	var action_to_take: int
	
	# Exploration vs Exploitation decision
	# Random action (explore) if random number < exploration_probability
	# Best known action (exploit) otherwise
	if randf() < exploration_probability and is_learning:
		action_to_take = randi_range(0, action_spaces - 1) # Random exploration
	else:
		action_to_take = QTable.index_of_max_from_row(current_state) # Exploit best known action
	
	# Q-Learning vs SARSA Update Rules:
    # Q-Learning uses max future Q-value: Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    # SARSA uses actual next action: Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
	if is_learning:
		# Q-value update only happens if we have a valid previous state
		if previous_state != -100:
			if not SARSA:
				# Q-Learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
				# Where α is learning_rate, γ is discounted_factor
				QTable.data[previous_state][previous_action] = (1 - learning_rate) * QTable.data[previous_state][previous_action] + \
				learning_rate * (reward_of_previous_state + discounted_factor * QTable.max_from_row(current_state))
			else:
				# SARSA update rule: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
				# Uses actual next action instead of maximum possible value
				QTable.data[previous_state][previous_action] = (1 - learning_rate) * QTable.data[previous_state][previous_action] + \
				learning_rate * (reward_of_previous_state + discounted_factor * QTable.data[current_state][action_to_take])
		
		# Update state/action memory for next iteration
		previous_state = current_state
		previous_action = action_to_take
		
		# Decay exploration probability if it's time to do so
		if decay_per_steps != 0:
			if steps_completed != 0 and steps_completed % decay_per_steps == 0:
				exploration_probability = maxf(min_exploration_probability, exploration_probability - exploration_decreasing_decay)
	
	# Debug output section
	if decay_per_steps != 0:
		if print_debug_info and steps_completed % decay_per_steps == 0:
			print("Total steps completed:", steps_completed) # Print total steps taken
			print("Current exploration probability:", exploration_probability) # Print current exploration probability
			print("Q-Table data:", QTable.data) # Print Q-table data
			print("-----------------------------------------------------------------------------------------")
	
	steps_completed += 1
	return action_to_take

# Serialization functions for saving/loading the trained model
func to_dict() -> Dictionary:
	var q_learning_dict = {
		"observation_space": observation_space,
		"action_spaces": action_spaces,
		"QTable": QTable.data,
		"SARSA": SARSA,
		"exploration_probability": exploration_probability,
		"exploration_decreasing_decay": exploration_decreasing_decay,
		"min_exploration_probability": min_exploration_probability,
		"discounted_factor": discounted_factor,
		"learning_rate": learning_rate,
		"decay_per_steps": decay_per_steps,
		"steps_completed": steps_completed,
		"previous_state": previous_state,
		"current_state": current_state,
		"previous_action": previous_action,
		"done": done,
		"is_learning": is_learning,
		"print_debug_info": print_debug_info
	}
	return q_learning_dict

func from_dict(q_learning_dict: Dictionary) -> void:
	observation_space = q_learning_dict["observation_space"]
	action_spaces = q_learning_dict["action_spaces"]
	QTable = Matrix.new(q_learning_dict["QTable"].size(), q_learning_dict["QTable"][0].size())
	SARSA = q_learning_dict["SARSA"]
	exploration_probability = q_learning_dict["exploration_probability"]
	exploration_decreasing_decay = q_learning_dict["exploration_decreasing_decay"]
	min_exploration_probability = q_learning_dict["min_exploration_probability"]
	discounted_factor = q_learning_dict["discounted_factor"]
	learning_rate = q_learning_dict["learning_rate"]
	decay_per_steps = q_learning_dict["decay_per_steps"]
	steps_completed = q_learning_dict["steps_completed"]
	previous_state = q_learning_dict["previous_state"]
	current_state = q_learning_dict["current_state"]
	previous_action = q_learning_dict["previous_action"]
	done = q_learning_dict["done"]
	is_learning = q_learning_dict["is_learning"]
	print_debug_info = q_learning_dict["print_debug_info"]

# File I/O operations for model persistence
func save(file_path: String) -> void:
	var file: FileAccess = FileAccess.open(file_path, FileAccess.WRITE)
	file.store_var(self.to_dict())
	file.close()

func load(file_path: String) -> void:
	var file: FileAccess = FileAccess.open(file_path, FileAccess.READ)
	var q_learning_dict: Dictionary = file.get_var()
	file.close()
	self.from_dict(q_learning_dict)