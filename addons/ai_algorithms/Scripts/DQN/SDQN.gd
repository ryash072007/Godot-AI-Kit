class_name SDQN

# Neural network parameters
var learning_rate: float = 0.01
var discount_factor: float = 0.99
var exploration_probability: float = 1.0
var min_exploration_probability: float = 0.01
var exploration_decay: float = 0.001
var batch_size: int = 32  # Number of transitions to sample for training
var max_episodes: int = 1000
var max_steps: int = 200

# Variables to hold state and action information
var state_space: int  # Number of state variables
var action_space: int  # Number of possible actions
var Q_network: NeuralNetworkAdvanced

var memory: Array[Array] = []
var steps: int = 0


func _init(state_space: int, action_space: int) -> void:
	self.state_space = state_space
	self.action_space = action_space

	Q_network = NeuralNetworkAdvanced.new()
	Q_network.add_layer(state_space)
	Q_network.add_layer(32, Q_network.ACTIVATIONS["RELU"])
	Q_network.add_layer(action_space, Q_network.ACTIVATIONS["LINEAR"])



func choose_action(state: Array) -> int:
	# Epsilon-greedy action selection
	if randf() < exploration_probability:
		return randi_range(0, action_space - 1)  # Explore
	else:
		return predict(state)  # Exploit


func predict(state: Array) -> int:
	var predicted_q_values: Array = Q_network.predict(state)
	var max_value_index: int = 0
	var max_value: float = predicted_q_values[max_value_index]
	for i in range(1, action_space):
		if predicted_q_values[i] > max_value:
			max_value_index = i
			max_value = predicted_q_values[max_value_index]
	return max_value_index


func max_q_predict(state: Array) -> float:
	var predicted_q_values: Array = Q_network.predict(state)
	var max_value: float = -INF
	for q_value in predicted_q_values:
		if q_value > max_value:
			max_value = q_value
	return max_value


func sample(array: Array) -> Array:
	var length: int = array.size()
	var indices: Array[int] = []
	var indices_length: int = 0
	var sample: Array = []
	while indices_length <= batch_size:
		var index: int = randi_range(0, length - 1)
		if index not in indices:
			indices.append(index)
			indices_length += 1
	for index in indices:
		sample.append(array[index])
	return sample


func train(replay_memory: Array) -> void:
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
		var target_q_values: Array = Q_network.predict(state)
		target_q_values[action] = target
		Q_network.train(state, target_q_values)

	memory.clear()


func add_memory(state: Array, action: int, reward: float, next_state: Array) -> void:
	memory.append([state, action, reward, next_state])
	steps += 1
	if steps >= max_steps:
		steps = 0
		exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decay)
		train(memory)

#func _process(delta: float) -> void:
	#for episode in range(max_episodes):
		#var state = get_initial_state()  # Implement your method to reset the environment and get the initial state
#
		#for step in range(max_steps):
			#var action = choose_action(state)
			#var (next_state, reward, done) = take_action(action)  # Implement your method to take action in the environment
#
			## Store transition in replay memory
			#replay_memory.append([state, action, reward, next_state])  # Simplified storage; consider limits on memory size
#
			## Train the Q-network
			#if replay_memory.size() >= batch_size:
				#train(replay_memory)
#
			#state = next_state
#
			#if done:
				#break  # Exit the loop if the episode is finished
#
		## Decay exploration probability
		#exploration_probability = max(min_exploration_probability, exploration_probability * exploration_decay)
#
		## Optional: Print progress or statistics
		#print("Episode:", episode, "Exploration probability:", exploration_probability)
