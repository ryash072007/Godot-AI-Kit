# Creation of Greaby (https://github.com/Greaby/godot-neuroevolution/blob/main/lib/neural_network.gd) from here till line 160
class_name NeuralNetwork

# Network architecture parameters
var best: bool = false
var input_nodes: int # Number of input neurons
var hidden_nodes: int # Number of hidden layer neurons
var output_nodes: int # Number of output neurons

# Weight matrices between layers
var weights_input_hidden: Matrix # Weights from input to hidden layer
var weights_hidden_output: Matrix # Weights from hidden to output layer

# Bias matrices for each layer
var bias_hidden: Matrix # Biases for hidden layer
var bias_output: Matrix # Biases for output layer

# Learning parameters
var learning_rate: float = 0.15 # Rate at which network learns

# Activation function settings
var ACTIVATIONS = Activation.new()
var activation_function: Callable # Function used for neuron activation
var activation_dfunction: Callable # Derivative of activation function

# Network performance tracking
var fitness: float = 0.0 # Fitness score for genetic algorithm

# Visualization
var color: Color = Color.TRANSPARENT # Color representation of network weights

# Input sensors
var raycasts: Array[RayCast2D] # Array of raycasts for environment sensing

# Initialize network with specified architecture
func _init(_input_nodes: int, _hidden_nodes: int, _output_nodes: int, is_set: bool = false) -> void:
	if !is_set:
		randomize()
		input_nodes = _input_nodes;
		hidden_nodes = _hidden_nodes;
		output_nodes = _output_nodes;

		weights_input_hidden = Matrix.rand(Matrix.new(hidden_nodes, input_nodes))
		weights_hidden_output = Matrix.rand(Matrix.new(output_nodes, hidden_nodes))

		bias_hidden = Matrix.rand(Matrix.new(hidden_nodes, 1))
		bias_output = Matrix.rand(Matrix.new(output_nodes, 1))

	set_activation_function()
	set_nn_color()

# Set network color based on weight values
func set_nn_color():
	color = Color(Matrix.average(weights_input_hidden),
	Matrix.average(weights_hidden_output),
	Matrix.average(Matrix.dot_product(bias_hidden, bias_output)), 1)

# Configure activation functions for neurons
func set_activation_function(callback: Callable = ACTIVATIONS.SIGMOID.function, dcallback: Callable = ACTIVATIONS.SIGMOID.derivative) -> void:
	activation_function = callback
	activation_dfunction = dcallback

# Forward propagation - predict output from input
func predict(input_array: Array[float]) -> Array:
	var inputs = Matrix.from_array(input_array)

	var hidden = Matrix.dot_product(weights_input_hidden, inputs)
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, activation_function)

	var output = Matrix.dot_product(weights_hidden_output, hidden)
	output = Matrix.add(output, bias_output)
	output = Matrix.map(output, activation_function)

	return Matrix.to_array(output)

# Backpropagation - train network using target values
func train(input_array: Array, target_array: Array):
	# Forward pass
	var inputs = Matrix.from_array(input_array)
	var targets = Matrix.from_array(target_array)

	var hidden = Matrix.dot_product(weights_input_hidden, inputs);
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, activation_function)

	var outputs = Matrix.dot_product(weights_hidden_output, hidden)
	outputs = Matrix.add(outputs, bias_output)
	outputs = Matrix.map(outputs, activation_function)

	# Calculate output layer errors
	var output_errors = Matrix.subtract(targets, outputs)

	# Update weights and biases for output layer
	var gradients = Matrix.map(outputs, activation_dfunction)
	gradients = Matrix.multiply(gradients, output_errors)
	gradients = Matrix.scalar(gradients, learning_rate)

	var hidden_t = Matrix.transpose(hidden)
	var weight_ho_deltas = Matrix.dot_product(gradients, hidden_t)

	weights_hidden_output = Matrix.add(weights_hidden_output, weight_ho_deltas)
	bias_output = Matrix.add(bias_output, gradients)

	# Calculate hidden layer errors
	var weights_hidden_output_t = Matrix.transpose(weights_hidden_output)
	var hidden_errors = Matrix.dot_product(weights_hidden_output_t, output_errors)

	# Update weights and biases for hidden layer
	var hidden_gradient = Matrix.map(hidden, activation_dfunction)
	hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors)
	hidden_gradient = Matrix.scalar(hidden_gradient, learning_rate)

	var inputs_t = Matrix.transpose(inputs)
	var weight_ih_deltas = Matrix.dot_product(hidden_gradient, inputs_t)

	weights_input_hidden = Matrix.add(weights_input_hidden, weight_ih_deltas)

	bias_hidden = Matrix.add(bias_hidden, hidden_gradient)


# Genetic algorithm functions
# Combine two neural networks to create offspring
static func reproduce(a: NeuralNetwork, b: NeuralNetwork) -> NeuralNetwork:
	var result = NeuralNetwork.new(a.input_nodes, a.hidden_nodes, a.output_nodes)
	result.weights_input_hidden = Matrix.random(a.weights_input_hidden, b.weights_input_hidden)
	result.weights_hidden_output = Matrix.random(a.weights_hidden_output, b.weights_hidden_output)
	result.bias_hidden = Matrix.random(a.bias_hidden, b.bias_hidden)
	result.bias_output = Matrix.random(a.bias_output, b.bias_output)

	return result

# Apply random mutations to network weights
static func mutate(nn: NeuralNetwork, callback: Callable = NeuralNetwork.mutate_callable_reproduced) -> NeuralNetwork:
	var result = NeuralNetwork.new(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.map(nn.weights_input_hidden, callback)
	result.weights_hidden_output = Matrix.map(nn.weights_hidden_output, callback)
	result.bias_hidden = Matrix.map(nn.bias_hidden, callback)
	result.bias_output = Matrix.map(nn.bias_output, callback)
	return result

# Mutation function with small changes
static func mutate_callable_reproduced(value, _row, _col):
	randomize()
	value += randf_range(-0.15, 0.15)
	return value

# Create exact copy of neural network
static func copy(nn: NeuralNetwork) -> NeuralNetwork:
	var result = NeuralNetwork.new(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.copy(nn.weights_input_hidden)
	result.weights_hidden_output = Matrix.copy(nn.weights_hidden_output)
	result.bias_hidden = Matrix.copy(nn.bias_hidden)
	result.bias_output = Matrix.copy(nn.bias_output)
	result.color = nn.color
	result.fitness = nn.fitness
	return result


# Creation of ryash072007 from here onwards

# Get sensor inputs from raycasts
func get_inputs_from_raycasts() -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")

	var _input_array: Array[float]

	for ray in raycasts:
		if is_instance_valid(ray): _input_array.push_front(get_distance(ray))

	return _input_array

# Get network prediction using raycast inputs
func get_prediction_from_raycasts(optional_val: Array = []) -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")

	var _array_ = get_inputs_from_raycasts()
	_array_.append_array(optional_val)
	return predict(_array_)

# Calculate distance from raycast collision
func get_distance(_raycast: RayCast2D):
	var distance: float = 0.0
	if _raycast.is_colliding():
		var origin: Vector2 = _raycast.global_transform.get_origin()
		var collision: Vector2 = _raycast.get_collision_point()

		distance = origin.distance_to(collision)
	else:
		distance = sqrt((pow(_raycast.target_position.x, 2) + pow(_raycast.target_position.y, 2)))
	return distance
