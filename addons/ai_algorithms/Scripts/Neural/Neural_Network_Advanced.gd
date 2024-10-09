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
var learning_rate: float = 0.5

# Array to store the structure of the network (number of nodes in each layer)
var layer_structure: Array[int] = []

# Function to add a layer to the network
func add_layer(nodes: int, activation: Dictionary = ACTIVATIONS.SIGMOID):
	# If there is already a layer, we need to add weights and biases for the new layer
	if layer_structure.size() != 0:
		var layer_data: Dictionary = {
			"weights": Matrix.rand(Matrix.new(nodes, layer_structure[-1])), # Randomly initialize weights
			"bias": Matrix.rand(Matrix.new(nodes, 1)), # Randomly initialize biases
			"activation": activation # Set activation function for this layer
		}
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
	var next_layer_errors: Matrix

	# Loop backward through the network layers
	for layer_index in range(network.size() - 1, -1, -1):
		var layer: Dictionary = network[layer_index]
		var layer_outputs: Matrix = outputs[layer_index] # Outputs of the current layer
		var layer_unactivated_output: Matrix = Matrix.transpose(unactivated_outputs[layer_index]) # Unactivated outputs (for gradients)

		# If it's the output layer
		if layer_index == network.size() - 1:
			var output_errors: Matrix = Matrix.subtract(expected_output, layer_outputs) # Calculate output errors
			next_layer_errors = output_errors # Set output errors as next layer's errors
			var gradients: Matrix = Matrix.map(layer_outputs, layer.activation.derivative) # Calculate gradients (derivative of activation)
			gradients = Matrix.multiply(gradients, output_errors) # Multiply by the error
			gradients = Matrix.scalar(gradients, learning_rate) # Multiply by learning rate

			# Calculate weight updates (weight delta)
			var weight_delta: Matrix
			if layer_index == 0:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(inputs)) # Use inputs if first layer
			else:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(outputs[layer_index - 1])) # Otherwise, use previous layer outputs

			# Update weights and biases
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, gradients)
		else:
			# For hidden layers
			var weights_hidden_output_t = Matrix.transpose(network[layer_index + 1].weights) # Transpose weights of next layer
			var hidden_errors = Matrix.dot_product(weights_hidden_output_t, next_layer_errors) # Calculate hidden layer errors
			next_layer_errors = hidden_errors # Set hidden errors as next layer's errors
			var hidden_gradient = Matrix.map(layer_outputs, layer.activation.derivative) # Calculate gradients for hidden layers
			hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors) # Multiply by hidden layer error
			hidden_gradient = Matrix.scalar(hidden_gradient, learning_rate) # Multiply by learning rate

			# Calculate weight updates for hidden layers
			var inputs_t: Matrix
			if layer_index != 0:
				inputs_t = Matrix.transpose(outputs[layer_index - 1]) # Use previous layer's outputs
			else:
				inputs_t = Matrix.transpose(inputs) # Use inputs for the first layer
			var weight_delta = Matrix.dot_product(hidden_gradient, inputs_t) # Calculate weight delta

			# Update weights and biases
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, hidden_gradient)


# Copy the NNA Completely
func copy() -> NeuralNetworkAdvanced:
	var copied_nna: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()
	copied_nna.network = network.duplicate(true)
	copied_nna.learning_rate = learning_rate
	copied_nna.layer_structure = layer_structure.duplicate(true)
	return copied_nna

