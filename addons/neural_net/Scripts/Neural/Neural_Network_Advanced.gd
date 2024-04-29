class_name NeuralNetworkAdvanced

"""
TODO:
	1. Continous Output Type -> Ongoing
	2. Discrete Output Type -> Not started
"""

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

	func dot_weights(inputs: Matrix) -> Matrix:
		# print("_______________________")
		# print("cols: ", weights.cols)
		# print("rows: ", inputs.rows)
		return Matrix.dot_product(weights, inputs) # Dimensions = (m x 1)
	
	func add_biases(inputsDotWeight: Matrix) -> Matrix:
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

func add_layer(nodes: int, activation_function: Dictionary=self.ACTIVATIONS["IDENTITY"]):
	var new_layer
	if len(layer_data) == 0:
		new_layer = Layer.new(nodes, 1, activation_function)
	else:
		new_layer = Layer.new(nodes, layer_data[ - 1].nodes, activation_function)
	new_layer.index = no_of_layers
	no_of_layers += 1
	layer_data.append(new_layer)

func predict(input: Array) -> Array:
	assert(len(input) == layer_data[0].nodes, "The input data has to have the same number of elements as there is nodes in the first layer")
	var layer_input: Matrix = Matrix.from_array(input)
	var prediction: Matrix = forward_propagation(layer_input)
	return Matrix.to_array(prediction)


func predict_matrix(input: Matrix) -> Matrix:
	assert(input.rows == layer_data[0].nodes)
	var prediction: Matrix = forward_propagation(input)
	return prediction


func forward_propagation(_layer_input: Matrix):
	var layer_input: Matrix = _layer_input
	for layer_index in range(1, no_of_layers):
		var current_layer: Layer = layer_data[layer_index]
		var weighted_inputs: Matrix = current_layer.dot_weights(layer_input)
		var weighted_inputs_biased: Matrix = current_layer.add_biases(weighted_inputs)
		var activated_weighted_inputs_biased: Matrix = Matrix.map(weighted_inputs_biased, current_layer.activation_function)
		layer_input = activated_weighted_inputs_biased
	return layer_input


func train(inputs: Array[Array], target_outputs: Array[Array]) -> void:
	assert(len(inputs) != 0 and len(target_outputs) != 0)
	assert(len(inputs) == len(target_outputs))
	
	var length_of_train: int = len(inputs)

	for index in range(length_of_train):
		var input: Matrix = Matrix.from_array(inputs[index])
		var predicted_output: Matrix = self.predict_matrix(input)
		var target_output: Matrix =  Matrix.from_array(target_outputs[index])
		var loss: Matrix # Also called delta
		for layer_index in range(no_of_layers, 1, -1): # Going back in layer_data excluding the first one
			var current_layer: Layer = layer_data[layer_index]
			loss = Matrix.subtract(target_output, predicted_output)
			var d_output: Matrix = Matrix.map(predicted_output, current_layer.d_activation_function)
			var d_outputXloss: Matrix = Matrix.multiply(d_output, loss) # (m x 1)
			var transposed_weights: Matrix = Matrix.transpose(current_layer.weights) # (n x m)
			var loss_previous_layer: Matrix = Matrix.dot_product(transposed_weights, d_outputXloss)

			loss = loss_previous_layer

