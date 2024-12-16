class_name SDQN

# Neural network parameters
var learning_rate: float = 0.001
var discount_factor: float = 0.95
var exploration_probability: float = 0.9
var min_exploration_probability: float = 0.2
var exploration_decay: float = 0.01
var batch_size: int = 128
var max_steps: int = 2048
var target_update_frequency: int = 4096  # Update target network every 4096 steps
var max_memory_size: int = 4096  # Max size of replay memory
var automatic_decay: bool = true

# Variables to hold state and action information
var state_space: int
var action_space: int
var Q_network: NeuralNetworkAdvanced
var target_Q_network: NeuralNetworkAdvanced

var memory: Array[Array] = []
var steps: int = 0
var update_steps: int = 0  # Counter for updating target network


func _init(state_space: int, action_space: int, learning_rate: float = 0.001) -> void:
	self.state_space = state_space
	self.action_space = action_space
	self.learning_rate = learning_rate

	Q_network = NeuralNetworkAdvanced.new()
	Q_network.add_layer(state_space)
	Q_network.add_layer(32, Q_network.ACTIVATIONS["ELU"])
	Q_network.add_layer(16, Q_network.ACTIVATIONS["ELU"])
	Q_network.add_layer(action_space, Q_network.ACTIVATIONS["LINEAR"])
	Q_network.learning_rate = learning_rate

	target_Q_network = Q_network.copy()


func choose_action(state: Array) -> int:
	# Epsilon-greedy action selection
	if randf() < exploration_probability:
		return randi_range(0, action_space - 1)  # Explore
	else:
		return predict(state)  # Exploit


func predict(state: Array) -> int:
	var predicted_q_values: Array = Q_network.predict(state)
	if NAN in predicted_q_values:
		print("NaN value detected -> exiting")
		#get_tree().quit()
	#print(predicted_q_values)
	var max_value_index: int = 0
	var max_value: float = predicted_q_values[max_value_index]
	for i in range(1, action_space):
		if predicted_q_values[i] > max_value:
			max_value_index = i
			max_value = predicted_q_values[max_value_index]
	return max_value_index


func max_q_predict(state: Array) -> float:
	var predicted_q_values: Array = target_Q_network.predict(state)
	var max_value: float = -INF
	for q_value in predicted_q_values:
		if q_value > max_value:
			max_value = q_value
	return max_value


func sample(array: Array) -> Array:
	var length: int = array.size()
	var indices: Array[int] = []
	var sample: Array = []

	 #Choose a random number of sequential elements (2-4 sequential elements)
#
	#var num_num_sequential = randi_range(0, 2)
#
	#for n in range(num_num_sequential):
		#var num_sequential = randi_range(4, 8)
#
		## Randomly choose a starting point for the sequential elements
		#var start_index = randi_range(0, length - num_sequential)
#
		## Add sequential elements to the indices
		#for i in range(num_sequential):
			#var index: int = (start_index + i) % length
			#if index not in indices:
				#indices.append(index)

	# Fill the rest with non-sequential random elements
	while indices.size() < batch_size:
		var index: int = randi_range(0, length - 1)
		if index not in indices:
			indices.append(index)

	# Build the final sample from the indices
	for index in indices:
		sample.append(array[index])

	return sample



func train(replay_memory: Array) -> void:
	print("Training Now")
	# Sample a minibatch from the replay memory
	var minibatch: Array = sample(replay_memory)

	for transition in minibatch:
		var state: Array = transition[0]
		var action: int = transition[1]
		var reward: int = transition[2]
		var next_state: Array = transition[3]

		# Calculate the target Q-value
		var target = reward + discount_factor * max_q_predict(next_state)  # Predict the max Q-value for next state

		# Update the Q-network
		if state.size() != state_space:
			print("Erronius state detected, skipping it!")
			continue
		var target_q_values: Array = Q_network.predict(state)
		target_q_values[action] = target
		Q_network.train(state, target_q_values)



func add_memory(state: Array, action: int, reward: float, next_state: Array) -> void:
	# Add new experience to memory
	memory.append([state, action, reward, next_state])

	# Limit memory size
	if memory.size() > max_memory_size:
		memory.pop_front()  # Remove oldest memory if memory size exceeds the limit

	# Increment step count and train
	steps += 1
	if steps >= max_steps:
		steps = 0
		if automatic_decay:
			exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decay)
		train(memory)

	update_steps += 1
	if update_steps >= target_update_frequency:
		update_steps = 0
		print("Copying QNetwork into Target Network now")
		target_Q_network = Q_network.copy()


func save(file_path: String) -> void:
	var file := FileAccess.open(file_path, FileAccess.WRITE)
	var data: Dictionary = {
		"learning_rate": self.learning_rate,
		"discount_factor": self.discount_factor,
		"state_space": self.state_space,
		"action_space": self.action_space,
		"Q_network": Q_network.network,
		"Q_network_structure": Q_network.layer_structure,
		"target_Q_network": target_Q_network.network,
		"target_Q_network_structure": target_Q_network.layer_structure
	}
	file.store_var(data, true)
	file.close()

static func load_sdqn(file_path: String) -> SDQN:
	var file := FileAccess.open(file_path, FileAccess.READ)
	var data: Dictionary = file.get_var(true)
	var sdqn: SDQN = SDQN.new(data["state_space"], data["action_space"], data["learning_rate"])
	sdqn.discount_factor = data["discount_factor"]
	sdqn.Q_network.network = data["Q_network"]
	sdqn.Q_network.layer_structure = data["Q_network_structure"]
	sdqn.target_Q_network.network = data["target_Q_network"]
	sdqn.target_Q_network.layer_structure = data["target_Q_network_structure"]
	file.close()

	return sdqn

