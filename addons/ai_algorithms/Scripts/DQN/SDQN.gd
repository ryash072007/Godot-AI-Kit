class_name SDQN

# Neural network parameters
var learning_rate: float = 0.001
var discount_factor: float = 0.95
var exploration_probability: float = 1
var min_exploration_probability: float = 0.01
var exploration_decay: float = 0.99
var batch_size: int = 512
var max_steps: int = 512
var target_update_frequency: int = 1536  # Update target network every 2048 steps
var max_memory_size: int = 60 * 60 * 4  # Max size of replay memory
var automatic_decay: bool = false

var lr_decay_rate: float = 1
var final_learning_rate: float = 0.0001

# Variables to hold state and action information
var state_space: int
var action_space: int
var Q_network: NeuralNetworkAdvanced
var target_Q_network: NeuralNetworkAdvanced

var memory: Array[Array] = []
var steps: int = 0
var update_steps: int = 0  # Counter for updating target network
var trainings_done: int = 0

var use_multi_threading: bool = false
var train_thread: Thread
var train_mutex: Mutex
var train_semaphore: Semaphore
var semaphore_exit: bool = false
var Q_network_multi: NeuralNetworkAdvanced

var frame_skip: int = 8
var frames_done: int = 0

func _init(state_space: int, action_space: int, learning_rate: float = learning_rate) -> void:
	self.state_space = state_space
	self.action_space = action_space


func use_threading() -> void:
	self.use_multi_threading = true
	self.train_thread = Thread.new()
	self.train_mutex = Mutex.new()
	self.train_semaphore = Semaphore.new()
	self.Q_network_multi = Q_network.copy(true)

	self.train_thread.start(multithreaded_train)


func set_Q_network(neural_network: NeuralNetworkAdvanced) -> void:
	self.Q_network = neural_network.copy(true)
	self.Q_network.learning_rate = self.learning_rate
	self.target_Q_network = self.Q_network.copy()

func set_clip_value(clip_value: float) -> void:
	self.Q_network.clip_value = clip_value


func set_lr_value(lr: float) -> void:
	self.learning_rate = lr
	self.Q_network.learning_rate = self.learning_rate

func update_lr_linearly() -> void:
		self.learning_rate = max(learning_rate * lr_decay_rate, final_learning_rate)
		self.Q_network.learning_rate = self.learning_rate

func choose_action(state: Array) -> int:
	# Epsilon-greedy action selection
	if randf() < self.exploration_probability:
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

	if length < batch_size:
		return array

	if length > max_memory_size:
		length = max_memory_size

	# Fill the rest with non-sequential random elements
	while indices.size() < batch_size:
		var index: int = randi_range(0, length - 1)
		if index not in indices:
			indices.append(index)

	# Build the final sample from the indices
	for i in indices:
		sample.append(array[i])

	return sample



func train(replay_memory: Array) -> void:
	# Sample a minibatch from the replay memory
	var minibatch: Array = sample(replay_memory)

	for transition in minibatch:
		var state: Array = transition[0]
		var action: int = transition[1]
		var reward: int = transition[2]
		var next_state: Array = transition[3]
		var done: bool = transition[4]

		# Calculate the target Q-value
		var target: float  # Predict the max Q-value for next state

		# might be wrong, fix later
		if done:
			target = -1
		else:
			target = reward + discount_factor * max_q_predict(next_state)

		# Update the Q-network
		if state.size() != state_space:
			print("Erronius state detected, skipping it!")
			continue
		var target_q_values: Array = Q_network.predict(state)
		target_q_values[action] = target
		Q_network.train(state, target_q_values)


func multithreaded_train() -> void:
	while true:
		train_semaphore.wait()
		if semaphore_exit:
			break
		print(str(trainings_done) + " multi-train now")
		trainings_done += 1
		# Sample a minibatch from the replay memory
		var minibatch: Array = sample(memory)

		for transition in minibatch:
			var state: Array = transition[0]
			var action: int = transition[1]
			var reward: int = transition[2]
			var next_state: Array = transition[3]
			var done: bool = transition[4]

			# Calculate the target Q-value
			var target: float  # Predict the max Q-value for next state

			# might be wrong, fix later
			if done:
				target = -1
			else:
				target = reward + discount_factor * max_q_predict(next_state)

			# Update the Q-network
			if state.size() != state_space:
				print("Erronius state detected, skipping it!")
				continue

			var target_q_values: Array = Q_network_multi.predict(state)
			target_q_values[action] = target

			Q_network_multi.train(state, target_q_values)

		Q_network = Q_network_multi.copy()

func add_memory(state: Array, action: int, reward: float, next_state: Array, done: bool) -> void:
	# Add new experience to memory
	memory.append([state, action, reward, next_state, done])

	# Limit memory size
	if memory.size() > max_memory_size:
		memory.pop_front()  # Remove oldest memory if memory size exceeds the limit

	# Increment step count and train
	steps += 1
	if steps >= max_steps:
		steps = 0
		if automatic_decay:
			exploration_probability = max(min_exploration_probability, exploration_probability * exploration_decay)

		if lr_decay_rate != 1.0:
			update_lr_linearly()

		if use_multi_threading:
			train_semaphore.post()
		else:
			train(memory)


	update_steps += 1
	if update_steps >= target_update_frequency:
		update_steps = 0
		print("Copying QNetwork into Target Network now")
		if use_multi_threading:
			train_mutex.lock()
		target_Q_network = Q_network.copy()
		if use_multi_threading:
			train_mutex.unlock()

func close_threading() -> void:
	if use_multi_threading:
		semaphore_exit = true
		train_thread.wait_to_finish()

func copy() -> SDQN:
	var copied_dqn: SDQN = self
	copied_dqn.Q_network = self.Q_network.copy()
	copied_dqn.target_Q_network = self.target_Q_network.copy()
	return copied_dqn
