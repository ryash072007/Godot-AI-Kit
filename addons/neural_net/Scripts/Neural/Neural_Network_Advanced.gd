class_name NeuralNetworkAdvanced

static var ACTIVATIONS: Dictionary = {
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
	"IDENTITY": {
		"function": Callable(Activation, "identity"),
		"derivative": Callable(Activation, "didentity")
	}
}

class Layer:
	var index: int
	var nodes: int
	var input_nodes: int

	var activation_function: Callable
	var d_activation_function: Callable

	# The input has dimensions of (n x 1)
	# m is the number of nodes of the layer
	var weights: Matrix # Dimensions = (m x n)
	var biases: Matrix # Dimensions = # Dimensions = (m x 1)

	func dot_weights(inputs: Matrix):
		return Matrix.dot_product(weights, inputs) # Dimensions = (m x 1)
	
	func add_biases(inputsDotWeight: Matrix):
		return Matrix.add(inputsDotWeight, biases) # Order: (m x 1)
	
	func _init(_nodes: int, _input_nodes: int, _activation_functions):
		# Setting up values
		nodes = _nodes
		input_nodes = _input_nodes
		# Initialising weights and biases with random values
		weights = Matrix.rand_value_matrix(nodes, input_nodes)
		biases = Matrix.rand_value_matrix(nodes, 1)
		# Setting activation functions
		activation_function = _activation_functions["function"]
		d_activation_function = _activation_functions["derivative"]

var no_of_layers: int = 0
var layer_data: Array[Layer] = []

func add_layer(nodes: int, activation_function: Dictionary=ACTIVATIONS["IDENTITY"]):
	var new_layer
	if len(layer_data) == 0:
		new_layer = Layer.new(nodes, 1, activation_function)
	else:
		new_layer = Layer.new(nodes, layer_data[-1].nodes, activation_function)
	new_layer.index = no_of_layers
	no_of_layers += 1
	layer_data.append(new_layer)

