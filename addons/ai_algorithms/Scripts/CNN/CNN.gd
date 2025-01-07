class_name CNN

var learning_rate: float = 0.005
var layers: Array = []

var labels: Dictionary = {}

class Layer:
	class MutliFilterConvolutional1D:
		var filters: Array[Matrix]
		var num_filters: int
		var biases: Matrix
		var stride: int
		var input_shape: Vector2i
		var filter_shape: Vector2i = Vector2i(3, 3)
		var padding: int
		var output_shape: Vector2i
		var padding_mode: String

		var inputs: Array

		var activationFunction := Activation.RELU

		func _init(_num_filters: int = 1,  _padding_mode: String = "valid", _filter_shape: Vector2i = Vector2i(3, 3), _stride: int = 1) -> void:
			filter_shape = _filter_shape
			num_filters = _num_filters
			stride = _stride
			padding_mode = _padding_mode

			biases = Matrix.new(num_filters, 1)

			for i in range(num_filters):
				filters.append(Matrix.uniform_he_init(Matrix.new(filter_shape.x, filter_shape.y), filter_shape.x * filter_shape.y))

		func set_input_shape(_input_shape: Vector2i) -> void:
			input_shape = _input_shape
			if padding_mode == "same":
				output_shape = Vector2i(
					ceil(float(input_shape.x) / float(stride)),
					ceil(float(input_shape.y) / float(stride))
				)
				padding = ((output_shape.x - 1) * stride + filter_shape.x - input_shape.x) / 2
			else:
				output_shape = Vector2i(
					((input_shape.x - filter_shape.x + 2 * padding) / stride) + 1,
					((input_shape.y - filter_shape.y + 2 * padding) / stride) + 1
				)

		func forward(_input: Array) -> Array[Matrix]:
			inputs = _input
			var outputs: Array[Matrix] = []
			for input in inputs:
				var output: Array[Matrix] = forward_single(input)
				for _output in output:
					outputs.append(_output)
			return outputs

		func forward_single(_input: Matrix) -> Array[Matrix]:
			var outputs: Array[Matrix] = []
			for filter_index in range(num_filters):
				var filter: Matrix = filters[filter_index]
				var bias: float = biases.data[filter_index][0]
				var _output: Matrix = Matrix.new(output_shape.x, output_shape.y)
				for i in range(0, input_shape.x - filter_shape.x + 1 + 2 * padding, stride):
					for j in range(0, input_shape.y - filter_shape.y + 1 + 2 * padding, stride):
						var sum: float = 0.0
						for x in range(filter_shape.x):
							for y in range(filter_shape.y):
								var input_x = i + x - padding
								var input_y = j + y - padding
								if input_x >= 0 and input_x < input_shape.x and input_y >= 0 and input_y < input_shape.y:
									sum += _input.data[input_x][input_y] * filter.data[x][y]
						_output.data[i / stride][j / stride] = sum + bias
				_output = Matrix.map(_output, activationFunction.function)
				outputs.append(_output)
			return outputs

		func backward(_dout: Array) -> Dictionary:
			var dB: Matrix = Matrix.new(num_filters, 1)
			var dW: Array[Matrix] = []
			var dX: Array[Matrix] = []
			for input_index in range(inputs.size()):
				var input: Matrix = inputs[input_index]
				var dout: Matrix = _dout[input_index]
				for filter_index in range(num_filters):
					var filter: Matrix = filters[filter_index]
					var _dW: Matrix = Matrix.new(filter_shape.x, filter_shape.y)
					var _dX: Matrix = Matrix.new(input_shape.x, input_shape.y)
					for i in range(0, input_shape.x - filter_shape.x + 1 + 2 * padding, stride):
						for j in range(0, input_shape.y - filter_shape.y + 1 + 2 * padding, stride):
							for x in range(filter_shape.x):
								for y in range(filter_shape.y):
									var input_x = i + x - padding
									var input_y = j + y - padding
									if input_x >= 0 and input_x < input_shape.x and input_y >= 0 and input_y < input_shape.y:
										_dW.data[x][y] += input.data[input_x][input_y] * dout.data[i / stride][j / stride]
										_dX.data[input_x][input_y] += filter.data[x][y] * dout.data[i / stride][j / stride]
							dB.data[filter_index][0] += dout.data[i / stride][j / stride]
					dW.append(_dW)
					dX.append(_dX)
			return {
				"dW": dW,
				"dB": dB,
				"dX": dX,
				"type": "MutliFilterConvolutional1D"
			}

	class MultiPoolPooling:
		var stride: int = 1
		var pool_size: Vector2i = Vector2i(0, 0)
		var input_shape: Vector2i = Vector2i(0, 0)
		var output_shape: Vector2i = Vector2i(0, 0)
		var num_pools: int
		var padding: int

		# Storing the input from the last forward pass
		var input: Array[Matrix]

		func _init(_padding: int = 0, _stride: int = 2, _pool_size: Vector2i = Vector2i(2, 2)) -> void:
			pool_size = _pool_size
			stride = _stride
			padding = _padding

		func set_input_shape(_input_shape: Vector2i, _num_pools: int) -> void:
			input_shape = _input_shape
			output_shape = Vector2i(
				((input_shape.x - pool_size.x + 2 * padding) / stride) + 1,
				((input_shape.y - pool_size.y + 2 * padding) / stride) + 1
			)
			num_pools = _num_pools

		# Max pooling
		func forward(input_array: Array[Matrix]) -> Array[Matrix]:
			input = input_array
			var outputs: Array[Matrix] = []
			for pool_index in range(num_pools):
				var _input: Matrix = input[pool_index]
				var output = Matrix.new(output_shape.x, output_shape.y)
				for i in range(0, input_shape.x - pool_size.x + 1, stride):
					for j in range(0, input_shape.y - pool_size.y + 1, stride):
						var max_val = -INF
						for x in range(pool_size.x):
							for y in range(pool_size.y):
								max_val = max(max_val, _input.data[i + x][j + y])
						output.data[i / stride][j / stride] = max_val
				outputs.append(output)
			return outputs

		func backward(dout: Array[Matrix]) -> Dictionary:
			var dX: Array[Matrix] = []
			for index in range(num_pools):
				var _input: Matrix = input[index]
				var _dout: Matrix = dout[index]
				var _dX = Matrix.new(input_shape.x, input_shape.y)
				for i in range(0, input_shape.x - pool_size.x + 1, stride):
					for j in range(0, input_shape.y - pool_size.y + 1, stride):
						var max_val = -INF
						var max_x = 0
						var max_y = 0
						for x in range(pool_size.x):
							for y in range(pool_size.y):
								if _input.data[i + x][j + y] > max_val:
									max_val = _input.data[i + x][j + y]
									max_x = i + x
									max_y = j + y
						_dX.data[max_x][max_y] = _dout.data[i / stride][j / stride]
				dX.append(_dX)
			return {
				"dX": dX,
				"type": "Pooling"
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

		func _init(_output_shape: int, _activation: String = "RELU") -> void:
			output_shape = Vector2i(_output_shape, 1)
			activation = _activation
			activationFunction = ACTIVATIONS.get(activation)
			biases = Matrix.new(output_shape.x, 1)

		func set_input_shape(_input_shape: Vector2i) -> void:
			input_shape = _input_shape.x

			if activation in ["RELU", "LEAKYRELU", "ELU", "LINEAR"]:
				weights = Matrix.uniform_he_init(Matrix.new(output_shape.x, input_shape), input_shape)
			elif activation in ["SIGMOID", "TANH"]:
				weights = Matrix.uniform_glorot_init(Matrix.new(output_shape.x, input_shape), input_shape, output_shape.x)
			else:
				weights = Matrix.rand(Matrix.new(output_shape.x, input_shape))

		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.add(Matrix.dot_product(weights, input), biases)
			output = Matrix.map(output, activationFunction.function)
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dW: Matrix = Matrix.outer_product(dout, input)
			var dB: Matrix = dout
			var dX: Matrix = Matrix.dot_product(Matrix.transpose(weights), dout)
			return {
				"dW": dW,
				"dB": dB,
				"dX": dX,
				"type": "Dense"
			}

	class Flatten:
		var input_shape: Vector2i
		var output_shape: Vector2i
		var output: Matrix
		var num_feature_maps: int

		# Storing the input from the last forward pass
		var input: Array[Matrix]

		func set_input_shape(_input_shape: Vector2i, _num_feature_maps: int) -> void:
			input_shape = _input_shape
			num_feature_maps = _num_feature_maps
			output_shape = Vector2i(input_shape.x * input_shape.y * num_feature_maps, 1)

		func forward(_input: Array[Matrix]) -> Matrix:
			input = _input
			output = Matrix.new(output_shape.x, output_shape.y)
			for feature_index in range(num_feature_maps):
				for i in range(input_shape.x):
					for j in range(input_shape.y):
						output.data[feature_index * input_shape.x * input_shape.y + i * input_shape.y + j][0] = input[feature_index].data[i][j]
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dX: Array[Matrix] = []
			for feature_index in range(num_feature_maps):
				var _dX = Matrix.new(input_shape.x, input_shape.y)
				for i in range(input_shape.x):
					for j in range(input_shape.y):
						_dX.data[i][j] = dout.data[feature_index * input_shape.x * input_shape.y + i * input_shape.y + j][0]
				dX.append(_dX)
			return {
				"dX": dX,
				"type": "Flatten"
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

		func _init(_output_shape: int) -> void:
			output_shape = Vector2i(_output_shape, 1)
			biases = Matrix.new(output_shape.x, 1)

		func set_input_shape(_input_shape: Vector2i) -> void:
			input_shape = _input_shape.x
			weights = Matrix.uniform_he_init(Matrix.new(output_shape.x, input_shape), input_shape)

		func forward(_input: Matrix) -> Matrix:
			input = _input
			output = Matrix.add(Matrix.dot_product(weights, input), biases)
			output = softmax(output)
			return output

		func backward(dout: Matrix) -> Dictionary:
			var dW: Matrix = Matrix.outer_product(dout, input)
			var dB: Matrix = dout
			var dX: Matrix = Matrix.dot_product(Matrix.transpose(weights), dout)
			return {
				"dW": dW,
				"dB": dB,
				"dX": dX,
				"type": "SoftmaxDense"
			}

		func softmax(x: Matrix) -> Matrix:
			var max_val: float = Matrix.max(x)
			var x_stable: Matrix = Matrix.scalar_add(x, -max_val)
			var exps: Matrix = Matrix.map(x_stable, func(value: float, _row: int, _col: int): return exp(value))
			var sum_exp: float = Matrix.sum(exps)
			return Matrix.scalar(exps, 1 / sum_exp)

	class BatchNormalization:
		var gamma: Matrix
		var beta: Matrix
		var epsilon: float = 1e-5
		var input_shape: Vector2i
		var output_shape: Vector2i
		var flattened: bool = false

		var input: Array[Matrix] = []
		var normalized: Array[Matrix] = []
		var mean: Matrix
		var variance: Matrix

		func set_input_shape(_input_shape: Vector2i, _flattened: bool = false) -> void:
			input_shape = _input_shape
			output_shape = _input_shape
			flattened = _flattened
			if flattened:
				gamma = Matrix.new(input_shape.x, 1, 1.0)
				beta = Matrix.new(input_shape.x, 1, 0.0)
			else:
				gamma = Matrix.new(input_shape.x, input_shape.y, 1.0)
				beta = Matrix.new(input_shape.x, input_shape.y, 0.0)

		func forward(_input):
			input = _input
			mean = Matrix.new(input_shape.x, input_shape.y)
			variance = Matrix.new(input_shape.x, input_shape.y)

			for matrix in input:
				mean = Matrix.add(mean, matrix)
			mean = Matrix.scalar(mean, 1.0 / input.size())

			for matrix in input:
				variance = Matrix.add(variance, Matrix.square(Matrix.subtract(matrix, mean)))
			variance = Matrix.scalar(variance, 1.0 / input.size())

			normalized.clear()
			for matrix in input:
				var norm: Matrix = Matrix.divide(Matrix.subtract(matrix, mean), Matrix.scalar_add(Matrix.square_root(variance), epsilon))
				normalized.append(Matrix.add(Matrix.multiply(norm, gamma), beta))

			return normalized

		func backward(dout: Array[Matrix]) -> Dictionary:
			var dgamma = Matrix.new(input_shape.x, input_shape.y)
			var dbeta = Matrix.new(input_shape.x, input_shape.y)
			var dX = []

			var N = input.size()
			for i in range(N):
				dgamma = Matrix.add(dgamma, Matrix.multiply(dout[i], normalized[i]))
				dbeta = Matrix.add(dbeta, dout[i])

			for i in range(N):
				var x_hat = Matrix.divide(Matrix.subtract(input[i], mean), Matrix.square_root(Matrix.scalar_add(variance, epsilon)))
				var dx_hat = Matrix.multiply(dout[i], gamma)

				var dvar: Matrix = Matrix.multiply(dx_hat, Matrix.subtract(input[i], mean))
				dvar = Matrix.multiply(dvar, Matrix.scalar_denominator(-0.5, Matrix.power(Matrix.scalar_add(Matrix.square_root(variance), epsilon), 3)))

				var dmean: Matrix = Matrix.multiply(dx_hat, Matrix.scalar_denominator(-1.0, Matrix.scalar_add(Matrix.square_root(variance), epsilon)))
				dmean = Matrix.add(dmean, Matrix.scalar(dvar, -2.0 / N))

				var dx: Matrix = Matrix.add(Matrix.multiply(dx_hat, Matrix.scalar_denominator(1.0, Matrix.scalar_add(Matrix.square_root(variance), epsilon))), Matrix.scalar(dvar, 2.0 / N))
				dx = Matrix.add(dx, Matrix.scalar(dmean, 1.0 / N))
				dX.append(dx)

			return {
				"dX": dX,
				"dgamma": dgamma,
				"dbeta": dbeta,
				"type": "BatchNormalization"
			}

	class Dropout:
		var rate: float
		var mask: Matrix
		var input_shape: Vector2i
		var flattened: bool

		func _init(_rate: float) -> void:
			rate = _rate

		func set_input_shape(_input_shape: Vector2i, _flattened: bool) -> void:
			input_shape = _input_shape
			flattened = _flattened

		func forward(_input):
			mask = Matrix.new(input_shape.x, input_shape.y)
			for i in range(input_shape.x):
				for j in range(input_shape.y):
					mask.data[i][j] = 0.0 if randf() < rate else 1.0
			
			if flattened:
				return Matrix.multiply(_input, mask)
			else:
				var outputs: Array[Matrix] = []
				for matrix in _input:
					outputs.append(Matrix.multiply(matrix, mask))
				return outputs

		func backward(dout) -> Dictionary:
			if flattened:
				return {
					"dX": Matrix.multiply(dout, mask),
					"type": "Dropout"
				}
			else:
				var dX: Array[Matrix] = []
				for i in range(dout.size()):
					dX.append(Matrix.multiply(dout[i], mask))
				return {
					"dX": dX,
					"type": "Dropout"
				}

func add_layer(layer) -> void:
	layers.append(layer)

func forward(input: Matrix, skip_dropout: bool = false) -> Matrix:
	var output = [input]
	for layer in layers:
		if skip_dropout and layer is Layer.Dropout:
			continue
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
	output_data.data[labels[label]][0] = 1.0

	# Forward pass
	var y_pred: Matrix = forward(input_data)
	var loss: float = cross_entropy_loss(y_pred, output_data)
	var grad = gradient_cross_entropy_loss(y_pred, output_data)

	# Backward pass and update weights
	for i in range(layers.size() - 1, -1, -1):
		var layer = layers[i]
		var gradients = layer.backward(grad)
		if gradients.has("dW"):
			if gradients["type"] == "MutliFilterConvolutional1D":
				for j in range(layer.num_filters):
					layer.filters[j] = Matrix.subtract(layer.filters[j], Matrix.scalar(gradients["dW"][j], learning_rate))
			else:
				layer.weights = Matrix.subtract(layer.weights, Matrix.scalar(gradients["dW"], learning_rate))
		if gradients.has("dB"):
			layer.biases = Matrix.subtract(layer.biases, Matrix.scalar(gradients["dB"], learning_rate))
		if gradients.has("dX"):
			grad = gradients["dX"]
		if gradients.has("dgamma"):
			layer.gamma = Matrix.subtract(layer.gamma, Matrix.scalar(gradients["dgamma"], learning_rate))
		if gradients.has("dbeta"):
			layer.beta = Matrix.subtract(layer.beta, Matrix.scalar(gradients["dbeta"], learning_rate))
	return loss

func categorise(input_data: Matrix):
	var y_pred: Matrix = forward(input_data, true)
	var max_index: int = Matrix.transpose(y_pred).index_of_max_from_row(0)
	for key in labels.keys():
		if labels[key] == max_index:
			return key
	return null

func add_labels(_labels: Array) -> void:
	for i in range(_labels.size()):
		labels[_labels[i]] = i

func compile_network(input_dimensions: Vector2i) -> void:
	var current_input_shape = input_dimensions
	var num_feature_maps = 1
	var flattened = false

	for layer in layers:
		if layer is Layer.MutliFilterConvolutional1D:
			layer.set_input_shape(current_input_shape)
			current_input_shape = layer.output_shape
			num_feature_maps *= layer.num_filters
			flattened = false
		elif layer is Layer.MultiPoolPooling:
			layer.set_input_shape(current_input_shape, num_feature_maps)
			current_input_shape = layer.output_shape
			flattened = false
		elif layer is Layer.Flatten:
			layer.set_input_shape(current_input_shape, num_feature_maps)
			current_input_shape = layer.output_shape
			flattened = true
		elif layer is Layer.Dense or layer is Layer.SoftmaxDense:
			layer.set_input_shape(current_input_shape)
			current_input_shape = layer.output_shape
		elif layer is Layer.BatchNormalization:
			layer.set_input_shape(current_input_shape, flattened)
		elif layer is Layer.Dropout:
			layer.set_input_shape(current_input_shape, flattened)


