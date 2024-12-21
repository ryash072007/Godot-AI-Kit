class_name NeuralNetworkAdvanced

# Array to store the network layers (weights, biases, and activations)
var network: Array

# Dictionary of activation functions and their corresponding derivatives
var ACTIVATIONS: Dictionary = {
	"SIGMOID": {
		"function": Callable(Activation, "sigmoid"),
		"derivative": Callable(Activation, "dsigmoid")
	},
	"RELU": {
		"function": Callable(Activation, "relu"),
		"derivative": Callable(Activation, "drelu")
	},
	"TANH": {
		"function": Callable(Activation, "tanh_"),
		"derivative": Callable(Activation, "dtanh")
	},
	"ARCTAN": {
		"function": Callable(Activation, "arcTan"),
		"derivative": Callable(Activation, "darcTan")
	},
	"PRELU": {
		"function": Callable(Activation, "prelu"),
		"derivative": Callable(Activation, "dprelu")
	},
	"ELU": {
		"function": Callable(Activation, "elu"),
		"derivative": Callable(Activation, "delu")
	},
	"SOFTPLUS": {
		"function": Callable(Activation, "softplus"),
		"derivative": Callable(Activation, "dsoftplus")
	},
	"LINEAR": {
		"function": Callable(Activation, "linear"),
		"derivative": Callable(Activation, "dlinear")
	}
}

# Learning rate for training the network
var learning_rate: float = 0.01

# Array to store the structure of the network (number of nodes in each layer)
var layer_structure: Array[int] = []

var clip_value: float = INF

# Adam optimiser
var use_adam_optimiser: bool = true
var beta1: float = 0.9
var beta2: float = 0.999
var epsilon: float = 1e-8
var m_weights: Array[Matrix] = [] # First moment for weights
var v_weights: Array[Matrix] = [] # Second moment for weights
var m_biases: Array[Matrix] = [] # First moment for biases
var v_biases: Array[Matrix] = [] # Second moment for biases
var t: int = 0 # Time step


func _init(use_adam: bool = true) -> void:
	self.use_adam_optimiser = use_adam


# Automatically considers type of function
# Function to add a layer to the network
func add_layer(nodes: int, activation: Dictionary = ACTIVATIONS.SIGMOID, use_optim_init: bool = true, random_biases: bool = false):
	# If there is already a layer, we need to add weights and biases for the new layer
	if layer_structure.size() != 0:

		var weights: Matrix
		var bias: Matrix

		if use_optim_init:
			if activation in [ACTIVATIONS.RELU, ACTIVATIONS.PRELU, ACTIVATIONS.ELU, ACTIVATIONS.LINEAR]:
				print("Using He init")
				weights = Matrix.uniform_he_init(Matrix.new(nodes, layer_structure[-1]), layer_structure[-1])
			elif activation in [ACTIVATIONS.SIGMOID, ACTIVATIONS.TANH]:
				print("Using Glorot init")
				weights = Matrix.uniform_glorot_init(Matrix.new(nodes, layer_structure[-1]), layer_structure[-1], nodes)
			else:
				print("Using rand init")
				weights = Matrix.rand(Matrix.new(nodes, layer_structure[-1]))
		else:
			print("Using rand init")
			weights = Matrix.rand(Matrix.new(nodes, layer_structure[-1]))

		if random_biases:
			bias = Matrix.rand(Matrix.new(nodes, 1))
		else:
			bias = Matrix.new(nodes, 1)

		var layer_data: Dictionary = {
			"weights": weights,
			"bias": bias,
			"activation": activation # Set activation function for this layer
		}

		if use_adam_optimiser:
			m_weights.append(Matrix.new(nodes, layer_structure[-1]))
			v_weights.append(Matrix.new(nodes, layer_structure[-1]))
			m_biases.append(Matrix.new(nodes, 1))
			v_biases.append(Matrix.new(nodes, 1))

		network.push_back(layer_data) # Add the layer to the network


	# Add the number of nodes to the layer structure
	layer_structure.append(nodes)

# Function to make a prediction with the neural network
func predict(input_array: Array) -> Array:
	# Convert input array to a matrix
	var inputs: Matrix = Matrix.from_array(input_array)
	# Forward pass through the network
	for layer in network:
		var product: Matrix = Matrix.dot_product(layer.weights, inputs) # Calculate the weighted sum of inputs
		var sum: Matrix = Matrix.add(product, layer.bias) # Add bias to the sum
		var map: Matrix = Matrix.map(sum, layer.activation.function) # Apply activation function
		inputs = map # Use the output of this layer as input for the next
	# Return the final output as an array
	return Matrix.to_array(inputs)

# Function to train the network using backpropagation
func train(input_array: Array, target_array: Array):
	# Convert input and target arrays to matrices
	var inputs: Matrix = Matrix.from_array(input_array)
	var targets: Matrix = Matrix.from_array(target_array)

	# Arrays to store outputs and unactivated outputs of each layer
	var layer_inputs: Matrix = inputs
	var outputs: Array[Matrix]
	var unactivated_outputs: Array[Matrix]

	# Forward pass through each layer
	for layer in network:
		var product: Matrix = Matrix.dot_product(layer.weights, layer_inputs) # Weighted sum of inputs
		var sum: Matrix = Matrix.add(product, layer.bias) # Add bias
		var map: Matrix = Matrix.map(sum, layer.activation.function) # Apply activation function
		layer_inputs = map # Set output as input for the next layer
		outputs.append(map) # Store the output of this layer
		unactivated_outputs.append(sum) # Store the unactivated output for later use

	# Start backpropagation by calculating output errors
	var expected_output: Matrix = targets
	var next_layer_errors: Matrix = null

	t += 1
	# Loop backward through the network layers
	for layer_index in range(network.size() - 1, -1, -1):
		var layer: Dictionary = network[layer_index]
		var layer_outputs: Matrix = outputs[layer_index]
		var layer_unactivated_output: Matrix = unactivated_outputs[layer_index]
		var current_error: Matrix
		# Determine current errors
		if next_layer_errors == null:
			# Output layer error
			current_error = Matrix.subtract(expected_output, layer_outputs)
		else:
			# Hidden layer error
			var weights_hidden_output_t = Matrix.transpose(network[layer_index + 1].weights)
			current_error = Matrix.dot_product(weights_hidden_output_t, next_layer_errors)
			current_error = Matrix.multiply(current_error, Matrix.map(layer_unactivated_output, layer.activation.derivative))

		# Gradient calculation
		var gradients: Matrix = Matrix.map(layer_outputs, layer.activation.derivative)
		gradients = Matrix.multiply(gradients, current_error)

		if clip_value != INF:
			gradients = Matrix.clamp_matrix(gradients, -clip_value, clip_value)

		# Weight updates
		var inputs_t: Matrix = Matrix.transpose(inputs) if layer_index == 0 else Matrix.transpose(outputs[layer_index - 1])
		var weight_delta: Matrix = Matrix.dot_product(gradients, inputs_t)

		# Update moments for weights
		if use_adam_optimiser:
			if layer_index >= m_weights.size():
				print("Errornius m_weights")
				return
			if layer_index >= v_weights.size():
				print("Errornius v_weights")
				return
			if layer_index >= m_biases.size():
				print("Errornius m_biases")
				return
			if layer_index >= v_biases.size():
				print("Errornius v_biases")
				return

			m_weights[layer_index] = Matrix.add(Matrix.scalar(m_weights[layer_index], beta1), Matrix.scalar(weight_delta, 1.0 - beta1))
			v_weights[layer_index] = Matrix.add(Matrix.scalar(v_weights[layer_index], beta2), Matrix.scalar(Matrix.square(weight_delta), 1.0 - beta2))

			# Bias updates
			m_biases[layer_index] = Matrix.add(Matrix.scalar(m_biases[layer_index], beta1), Matrix.scalar(current_error, 1.0 - beta1))
			v_biases[layer_index] = Matrix.add(Matrix.scalar(v_biases[layer_index], beta2), Matrix.scalar(Matrix.square(current_error), 1.0 - beta2))

			# Compute bias-corrected first and second moment estimates
			var m_hat_weight: Matrix = Matrix.scalar(m_weights[layer_index], 1 / (1 - pow(beta1, t)))
			var v_hat_weight: Matrix = Matrix.scalar(v_weights[layer_index], 1 / (1 - pow(beta2, t)))
			var m_hat_bias: Matrix = Matrix.scalar(m_biases[layer_index], 1 / (1 - pow(beta1, t)))
			var v_hat_bias: Matrix = Matrix.scalar(v_biases[layer_index], 1 / (1 - pow(beta2, t)))

			var denominator_weight: Matrix = Matrix.add(Matrix.square_root(v_hat_weight), Matrix.new(v_hat_weight.rows, v_hat_weight.cols, epsilon))
			var denominator_bias: Matrix = Matrix.add(Matrix.square_root(v_hat_bias), Matrix.new(v_hat_bias.rows, v_hat_bias.cols, epsilon))

			# Update weights and biases using Adam
			network[layer_index].weights = Matrix.subtract(layer.weights, Matrix.divide(Matrix.scalar(m_hat_weight, learning_rate), denominator_weight))
			network[layer_index].bias = Matrix.subtract(layer.bias, Matrix.divide(Matrix.scalar(m_hat_bias, learning_rate), denominator_bias))
		else:
			gradients = Matrix.scalar(gradients, learning_rate)
			# Update weights and biases
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, gradients)

		# Pass current error to the next layer
		next_layer_errors = current_error


# Copy the NNA Completely
func copy() -> NeuralNetworkAdvanced:
	var copied_nna: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()
	copied_nna.network = network.duplicate(true)
	copied_nna.learning_rate = learning_rate
	copied_nna.layer_structure = layer_structure.duplicate(true)
	return copied_nna
