extends Node
class_name NeuralNetwork

@export var input_nodes: int
@export var hidden_nodes: int 
@export var output_nodes: int

var weights_input_hidden: Matrix
var weights_hidden_output: Matrix

var bias_hidden: Matrix
var bias_output: Matrix

@export var learning_rate: float = 0.15

var activation_function: Callable
var activation_dfunction: Callable

var fitness: float = 0.0

@export var color: Color = Color(0, 0, 0, 0)

@export var raycasts: Array[RayCast2D]

var nodes_set: bool = false

func _init() -> void:
	set_activation_function()

func set_nn_data(_input_nodes: int, _hidden_nodes: int, _output_nodes: int) -> void:
	assert(_input_nodes != 0 or _output_nodes != 0, "The NN's input or output nodes can NOT be set to 0!")
	
	randomize()
	
	input_nodes = _input_nodes;
	hidden_nodes = _hidden_nodes;
	output_nodes = _output_nodes;
	
	weights_input_hidden = Matrix.rand(Matrix.new(hidden_nodes, input_nodes))
	weights_hidden_output = Matrix.rand(Matrix.new(output_nodes, hidden_nodes))
	
	bias_hidden = Matrix.rand(Matrix.new(hidden_nodes, 1))
	bias_output = Matrix.rand(Matrix.new(output_nodes, 1))
	
	set_random_color()
	
	nodes_set = true

func set_nodes(_input_nodes: int, _hidden_nodes: int, _output_nodes: int) -> void:
	assert(_input_nodes != 0 or _output_nodes != 0, "The NN's input or output nodes can NOT be set to 0!")
	
	input_nodes = _input_nodes;
	hidden_nodes = _hidden_nodes;
	output_nodes = _output_nodes;
	
	set_random_color()

func set_random_color():
	color = Color(randi_range(0, 1), randi_range(0, 1), randi_range(0, 1), 1)

func set_activation_function(callback: Callable = Callable(Activation, "sigmoid"), dcallback: Callable = Callable(Activation, "dsigmoid")) -> void:
	activation_function = callback
	activation_dfunction = dcallback

func predict(input_array: Array) -> Array:
	var inputs = Matrix.from_array(input_array)
	
	var hidden = Matrix.product(weights_input_hidden, inputs)
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, activation_function)

	var output = Matrix.product(weights_hidden_output, hidden)
	output = Matrix.add(output, bias_output)
	output = Matrix.map(output, activation_function)

	return Matrix.to_array(output)

func train(input_array: Array, target_array: Array):
	var inputs = Matrix.from_array(input_array)
	var targets = Matrix.from_array(target_array)
	
#	print_debug(weights_input_hidden.data)
#	print_debug(inputs.data)
	
	var hidden = Matrix.product(weights_input_hidden, inputs);
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, activation_function)
	
	var outputs = Matrix.product(weights_hidden_output, hidden)
	outputs = Matrix.add(outputs, bias_output)
	outputs = Matrix.map(outputs, activation_function)
	
	var output_errors = Matrix.subtract(targets, outputs)
	
	var gradients = Matrix.map(outputs, activation_dfunction)
	gradients = Matrix.multiply(gradients, output_errors)
	gradients = Matrix.scalar(gradients, learning_rate)
	
	var hidden_t = Matrix.transpose(hidden)
	var weight_ho_deltas = Matrix.product(gradients, hidden_t)
	
	weights_hidden_output = Matrix.add(weights_hidden_output, weight_ho_deltas)
	bias_output = Matrix.add(bias_output, gradients)
	
	var weights_hidden_output_t = Matrix.transpose(weights_hidden_output)
	var hidden_errors = Matrix.product(weights_hidden_output_t, output_errors)
	
	var hidden_gradient = Matrix.map(hidden, activation_dfunction)
	hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors)
	hidden_gradient = Matrix.scalar(hidden_gradient, learning_rate)
	
	var inputs_t = Matrix.transpose(inputs)
	var weight_ih_deltas = Matrix.product(hidden_gradient, inputs_t)

	weights_input_hidden = Matrix.add(weights_input_hidden, weight_ih_deltas)

	bias_hidden = Matrix.add(bias_hidden, hidden_gradient)

func get_inputs_from_raycasts() -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")
	
	var _input_array: Array[float]
	
	for ray in raycasts:
		if is_instance_valid(ray): _input_array.push_front(get_distance(ray))
	
	return _input_array

func get_prediction_from_raycasts(optional_val: Array = []) -> Array[float]:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")
	
	var _array_ = get_inputs_from_raycasts()
	_array_.append_array(optional_val)
	return predict(_array_)

func get_distance(_raycast: RayCast2D):
	var distance: float = 0.0
	if _raycast.is_colliding():
		var origin: Vector2 = _raycast.global_transform.get_origin()
		var collision: Vector2 = _raycast.get_collision_point()
		
		distance = origin.distance_to(collision)

	return distance

static func reproduce(a: NeuralNetwork, b: NeuralNetwork) -> NeuralNetwork:
	var result = NeuralNetwork.new()
	result.set_nodes(a.input_nodes, a.hidden_nodes, a.output_nodes)
	result.weights_input_hidden = Matrix.random(a.weights_input_hidden, b.weights_input_hidden)
	result.weights_hidden_output = Matrix.random(a.weights_hidden_output, b.weights_hidden_output)
	result.bias_hidden = Matrix.random(a.bias_hidden, b.bias_hidden)
	result.bias_output = Matrix.random(a.bias_output, b.bias_output)

	return result

static func mutate(nn: NeuralNetwork, callback: Callable = Callable(NeuralNetwork, "mutate_callable")) -> NeuralNetwork:
	var result = NeuralNetwork.new()
	result.set_nodes(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.map(nn.weights_input_hidden, callback)
	result.weights_hidden_output = Matrix.map(nn.weights_hidden_output, callback)
	result.bias_hidden = Matrix.map(nn.bias_hidden, callback)
	result.bias_output = Matrix.map(nn.bias_output, callback)
	return result

static func copy(nn : NeuralNetwork) -> NeuralNetwork:
	var result = NeuralNetwork.new()
	result.set_nodes(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.copy(nn.weights_input_hidden)
	result.weights_hidden_output = Matrix.copy(nn.weights_hidden_output)
	result.bias_hidden = Matrix.copy(nn.bias_hidden)
	result.bias_output = Matrix.copy(nn.bias_output)
	return result

static func mutate_callable(value, _row, _col):
	seed(randi())
	randomize()
#	if randf_range(0, 1) < 0.7:
	value += randf_range(-0.1, 0.1)
		
	return value
