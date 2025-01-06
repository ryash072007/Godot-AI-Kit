class_name CNN

var learning_rate: float = 0.01
var layers: Array = []

var labels: Dictionary = {}

class Layer:
	class SingleFilterConvolutional1D:
		var filter: Matrix
		var bias: float = 0.0
		var stride: int = 1
		var input_shape: Vector2i
		var filter_shape: Vector2i = Vector2i(3, 3)
		var output_shape: Vector2i
		var output: Matrix

		var input: Matrix

		var activationFunction := Activation.RELU

		func _init(_input_shape: Vector2i, _filter_shape: Vector2i = Vector2i(3, 3), _stride: int = 1) -> void:
			input_shape = _input_shape
			filter_shape = _filter_shape

			output_shape = Vector2i(
				((input_shape.x - filter_shape.x) / stride) + 1,
				((input_shape.y - filter_shape.y) / stride) + 1
			)

			stride = _stride
			filter = Matrix.uniform_he_init(Matrix.new(filter_shape.x, filter_shape.y), filter_shape.x * filter_shape.y)

		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.new(output_shape.x, output_shape.y)
			for i in range(0, input_shape.x - filter_shape.x + 1, stride):
				for j in range(0, input_shape.y - filter_shape.y + 1, stride):
					var sum = 0.0
					for x in range(filter_shape.x):
						for y in range(filter_shape.y):
							sum += input.data[i + x][j + y] * filter.data[x][y]
					output.data[i / stride][j / stride] = sum + bias
			output = Matrix.map(output, activationFunction.function)
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dW: Matrix = Matrix.new(filter_shape.x, filter_shape.y)
			var dB: float = 0.0
			var dX: Matrix = Matrix.new(input_shape.x, input_shape.y)
			for i in range(0, input_shape.x - filter_shape.x + 1, stride):
				for j in range(0, input_shape.y - filter_shape.y + 1, stride):
					for x in range(filter_shape.x):
						for y in range(filter_shape.y):
							dW.data[x][y] += input.data[i + x][j + y] * dout.data[i / stride][j / stride]
							dX.data[i + x][j + y] += filter.data[x][y] * dout.data[i / stride][j / stride]
					dB += dout.data[i / stride][j / stride]
			return {
				"dW": dW,
				"dB": dB,
				"dX": dX
			}

	class Pooling:
		var stride: int = 1
		var pool_size: Vector2i = Vector2i(0, 0)
		var input_shape: Vector2i = Vector2i(0, 0)
		var output_shape: Vector2i = Vector2i(0, 0)
		var output: Matrix

		# Storing the input from the last forward pass
		var input: Matrix

		func _init(_input_shape: Vector2i, _pool_size: Vector2i = Vector2i(2, 2), _stride: int = 2) -> void:
			input_shape = _input_shape
			pool_size = _pool_size
			stride = _stride

			output_shape = Vector2i(
				((input_shape.x - pool_size.x) / stride) + 1,
				((input_shape.y - pool_size.y) / stride) + 1
			)

		# Max pooling
		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.new(output_shape.x, output_shape.y)
			for i in range(0, input_shape.x - pool_size.x + 1, stride):
				for j in range(0, input_shape.y - pool_size.y + 1, stride):
					var max_val = -INF
					for x in range(pool_size.x):
						for y in range(pool_size.y):
							max_val = max(max_val, input.data[i + x][j + y])
					output.data[i / stride][j / stride] = max_val
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dX = Matrix.new(input_shape.x, input_shape.y)
			for i in range(0, input_shape.x - pool_size.x + 1, stride):
				for j in range(0, input_shape.y - pool_size.y + 1, stride):
					var max_val = -INF
					var max_x = 0
					var max_y = 0
					for x in range(pool_size.x):
						for y in range(pool_size.y):
							if input.data[i + x][j + y] > max_val:
								max_val = input.data[i + x][j + y]
								max_x = i + x
								max_y = j + y
					dX.data[max_x][max_y] = dout.data[i / stride][j / stride]
			return {
				"dX": dX
			}

	class Dense:
		var weights: Matrix
		var biases: Matrix
		var activation: String = "RELU"
		var input_shape: int = 0
		var output_shape: Vector2i
		var output: Matrix

		# Storing the input from the last forward pass
		var input: Matrix

		var ACTIVATIONS: Activation = Activation.new()
		var activationFunction

		func _init(_input_shape: Vector2i, _output_shape: int, _activation: String = "RELU") -> void:
			input_shape = _input_shape.x
			output_shape = Vector2i(_output_shape, 1)
			activation = _activation

			activationFunction = ACTIVATIONS.get(activation)

			if activation in ["RELU", "LEAKYRELU", "ELU", "LINEAR"]:
				weights = Matrix.uniform_he_init(Matrix.new(output_shape.x, input_shape), input_shape)
			elif activation in ["SIGMOID", "TANH"]:
				weights = Matrix.uniform_glorot_init(Matrix.new(output_shape.x, input_shape), input_shape, output_shape.x)
			else:
				weights = Matrix.rand(Matrix.new(output_shape.x, input_shape))

			biases = Matrix.new(output_shape.x, 1)

		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.add(Matrix.dot_product(weights, input), biases)
			output = Matrix.map(output, activationFunction.function)
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dW: Matrix = Matrix.outer_product(dout, input)
			var dB: Matrix = dout
			var dX: Matrix = Matrix.multiply(weights, dout)
			return {
				"dW": dW,
				"dB": dB,
				"dX": dX
			}

	class Flatten:
		var input_shape: Vector2i
		var output_shape: Vector2i
		var output: Matrix

		# Storing the input from the last forward pass
		var input: Matrix

		func _init(_input_shape: Vector2i) -> void:
			input_shape = _input_shape
			output_shape = Vector2i(input_shape.x * input_shape.y, 1)

		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.new(output_shape.x, output_shape.y)
			for i in range(input_shape.x):
				for j in range(input_shape.y):
					output.data[i * input_shape.y + j][0] = input.data[i][j]
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dX = Matrix.new(input_shape.x, input_shape.y)
			for i in range(input_shape.x):
				for j in range(input_shape.y):
					dX.data[i][j] = dout.data[i * input_shape.y + j][0]
			return {
				"dX": dX
			}
    
	class SoftmaxDense:
		var weights: Matrix
		var biases: Matrix
		var activation: String = "SOFTMAX"
		var input_shape: int = 0
		var output_shape: Vector2i
		var output: Matrix

		# Storing the input from the last forward pass
		var input: Matrix

		func _init(_input_shape: Vector2i, _output_shape: int) -> void:
			input_shape = _input_shape.x
			output_shape = Vector2i(_output_shape, 1)

			weights = Matrix.uniform_glorot_init(Matrix.new(output_shape.x, input_shape), input_shape, output_shape.x)
			biases = Matrix.new(output_shape.x, 1)

		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.add(Matrix.dot_product(weights, input), biases)
			output = softmax(output)
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dW: Matrix = Matrix.outer_product(dout, input)
			var dB: Matrix = dout
			var dX: Matrix = Matrix.multiply(weights, dout)
			return {
				"dW": dW,
				"dB": dB,
				"dX": dX
			}
        
		func softmax(x: Matrix) -> Matrix:
			var max_val: float = x.data.max()
			var x_stable: Matrix = Matrix.scalar_add(x, -max_val)
			var exps: Matrix = Matrix.map(x_stable, func(value: float, _row: int, _col: int): return exp(value))
			var sum_exp: float = Matrix.sum(exps)
			return Matrix.scalar(exps, 1 / sum_exp)


func add_layer(layer) -> Vector2i:
	layers.append(layer)
	return layer.output_shape

func forward(input: Matrix) -> Matrix:
	var output = input
	for layer in layers:
		output = layer.forward(output)
	return output

func cross_entropy_loss(y_pred: Matrix, y_true: Matrix) -> float:
	var loss: float = 0.0
	var epsilon: float = 1e-12
	y_pred = Matrix.map(y_pred, func(value: float, _row: int, _col: int): return log(value + epsilon))
	var loss_matrix: Matrix = Matrix.multiply(y_true, y_pred)
	loss = -Matrix.sum(loss_matrix)
	return loss

func gradient_cross_entropy_loss(y_pred: Matrix, y_true: Matrix) -> Matrix:
	return Matrix.subtract(y_pred, y_true)

func train(input_data: Matrix, label) -> float:
	# One-hot encode the label
	var output_data: Matrix = Matrix.new(layers[-1].output_shape.x, 1)
	output_data.data[label][0] = 1.0

	var y_pred: Matrix = forward(input_data)
	var loss: float = cross_entropy_loss(y_pred, output_data)
	var grad: Matrix = gradient_cross_entropy_loss(y_pred, output_data)
	for i in range(layers.size() - 1, -1, -1):
		var layer = layers[i]
		var gradients = layer.backward(grad)
		if gradients.has("dW"):
			layer.weights = Matrix.subtract(layer.weights, Matrix.scalar(gradients["dW"], learning_rate))
		if gradients.has("dB"):
			layer.biases = Matrix.subtract(layer.biases, Matrix.scalar(gradients["dB"], learning_rate))
		if gradients.has("dX"):
			grad = gradients["dX"]
	return loss

func categorise(input_data: Matrix):
	var y_pred: Matrix = forward(input_data)
	var max_index: int = Matrix.transpose(y_pred).index_of_max_from_row(0)
	for key in labels.keys():
		if labels[key] == max_index:
			return key
	return null
	

func add_labels(_labels: Array) -> void:
	for i in range(_labels.size()):
		labels[_labels[i]] = i