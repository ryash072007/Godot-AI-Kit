class_name SDQN

# Neural network parameters
var learning_rate: float = 0.01
var discount_factor: float = 0.99
var exploration_probability: float = 0.9
var min_exploration_probability: float = 0.05
var exploration_decay: float = 0.99999
var batch_size: int = 96
var max_steps: int = 64
var target_update_frequency: int = 2048  # Update target network every 4096 steps
var max_memory_size: int = 8192  # Max size of replay memory
var automatic_decay: bool = false

var total_lr_decay_steps: int = 0 # 512 * 350 # max steps * no of training episodes
var initial_learning_rate: float = 0.01
var final_learning_rate: float = 0.0001

# Variables to hold state and action information
var state_space: int
var action_space: int
var Q_network: NeuralNetworkAdvanced
var target_Q_network: NeuralNetworkAdvanced

var memory: Array[Array] = []
var steps: int = 0
var update_steps: int = 0  # Counter for updating target network
var lr_decay_steps: int = 0


func _init(state_space: int, action_space: int, learning_rate: float = learning_rate) -> void:
	self.state_space = state_space
	self.action_space = action_space
	self.learning_rate = learning_rate

	Q_network = NeuralNetworkAdvanced.new()
	Q_network.add_layer(state_space)
	Q_network.add_layer(8, Q_network.ACTIVATIONS["ELU"])
	Q_network.add_layer(8, Q_network.ACTIVATIONS["ELU"])
	Q_network.add_layer(action_space, Q_network.ACTIVATIONS["LINEAR"])
	Q_network.learning_rate = learning_rate

	target_Q_network = Q_network.copy()


func set_clip_value(clip_value: float) -> void:
	Q_network.clip_value = clip_value


func update_lr_linearly() -> void:
	if lr_decay_steps <= total_lr_decay_steps:
		self.learning_rate = lerpf(initial_learning_rate, final_learning_rate, float(lr_decay_steps) / float(total_lr_decay_steps))
		self.Q_network.learning_rate = self.learning_rate

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

	var sample_size: int = min(batch_size, len(memory))

	# Fill the rest with non-sequential random elements
	while indices.size() < sample_size:
		var index: int = randi_range(0, length - 1)
		if index not in indices:
			indices.append(index)

	# Build the final sample from the indices
	for i in indices:
		sample.append(array[i])

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
			exploration_probability = max(min_exploration_probability, exploration_probability * exploration_decay)
		train(memory)

	lr_decay_steps += 1
	if total_lr_decay_steps != 0:
		update_lr_linearly()

	update_steps += 1
	if update_steps >= target_update_frequency:
		update_steps = 0
		print("Copying QNetwork into Target Network now")
		target_Q_network = Q_network.copy()


func copy() -> SDQN:
	var copied_dqn: SDQN = self
	copied_dqn.Q_network = self.Q_network.copy()
	copied_dqn.target_Q_network = self.target_Q_network.copy()
	return copied_dqn
