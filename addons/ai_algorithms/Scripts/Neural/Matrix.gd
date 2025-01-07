# Creation of Greaby (https://github.com/Greaby/godot-neuroevolution/blob/main/lib/matrix.gd) from here till line 124
class_name Matrix

var rows: int
var cols: int

var data: Array = []

func _init(_rows: int, _cols: int, value: float = 0.0) -> void:
	randomize()
	rows = _rows
	cols = _cols
	for row in range(rows):
		data.insert(row, [])
		for col in range(cols):
			data[row].insert(col, value)

static func from_array(arr: Array) -> Matrix:
	var result = Matrix.new(arr.size(), 1)
	for row in range(result.rows):
		result.data[row][0] = arr[row]
	return result

static func to_array(matrix: Matrix) -> Array:
	var result = []
	for row in range(matrix.rows):
		for col in range(matrix.cols):
			result.append(matrix.data[row][col])
	return result

static func rand(matrix: Matrix) -> Matrix:
	var result = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = randf_range(-0.15, 0.15)
	return result


static func add(a: Matrix, b: Matrix) -> Matrix:
	assert(a.rows == b.rows and a.cols == b.cols)

	var result = Matrix.new(a.rows, a.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = a.data[row][col] + b.data[row][col]

	return result

static func subtract(a: Matrix, b: Matrix) -> Matrix:
	assert(a.rows == b.rows and a.cols == b.cols)

	var result = Matrix.new(a.rows, a.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = a.data[row][col] - b.data[row][col]

	return result

static func scalar(matrix: Matrix, value: float) -> Matrix:
	var result = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = matrix.data[row][col] * value

	return result


static func dot_product(a: Matrix, b: Matrix) -> Matrix:
	assert(a.cols == b.rows)

	var result = Matrix.new(a.rows, b.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = 0.0
			for k in range(a.cols):
				result.data[row][col] += a.data[row][k] * b.data[k][col]

	return result


static func multiply(a: Matrix, b: Matrix) -> Matrix:
	assert(a.rows == b.rows and a.cols == b.cols)

	var result = Matrix.new(a.rows, a.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = a.data[row][col] * b.data[row][col]

	return result

static func copy(matrix: Matrix) -> Matrix:
	var result = Matrix.new(matrix.rows, matrix.cols)
	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = matrix.data[row][col]
	return result


static func transpose(matrix: Matrix) -> Matrix:
	var result = Matrix.new(matrix.cols, matrix.rows)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = matrix.data[col][row]

	return result

static func map(matrix: Matrix, callback: Callable) -> Matrix:
	var result = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = callback.call(matrix.data[row][col], row, col)

	return result


# Sole creation of ryash072007 from here onwards

static func clamp_matrix(matrix: Matrix, lower_clamp: float, upper_clamp: float) -> Matrix:
	var result = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = clampf(result.data[row][col], lower_clamp, upper_clamp)

	return result

static func random(a: Matrix, b: Matrix) -> Matrix:
	var result = Matrix.new(a.rows, a.cols)
	for row in range(result.rows):
		for col in range(result.cols):
			randomize()
			var _random = randf_range(0, 1)
			result.data[row][col] = a.data[row][col] if _random > 0.5 else b.data[row][col]

	return result


static func average(matrix: Matrix) -> float:
	var average_value: float = 0.0
	for row in range(matrix.rows):
		for col in range(matrix.cols):
			average_value += matrix.data[row][col]
	average_value = average_value / matrix.rows * matrix.cols
	return average_value

static func variance(matrix: Matrix) -> float:
	var mean_value = Matrix.average(matrix)
	var sum_value: float = 0.0
	for row in range(matrix.rows):
		for col in range(matrix.cols):
			sum_value += pow(matrix.data[row][col] - mean_value, 2)
	return sum_value / (matrix.rows * matrix.cols)

func index_of_max_from_row(_row: int) -> int:
	return data[_row].find(data[_row].max())

func max_from_row(_row: int) -> float:
	return data[_row].max()

static func max_matrix(a: Matrix, b: Matrix) -> Matrix:
	assert(a.rows == b.rows and a.cols == b.cols)

	var result: Matrix = Matrix.new(a.rows, b.cols)
	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = max(a.data[row][col], b.data[row][col])

	return result

static func min_matrix(a: Matrix, b: Matrix) -> Matrix:
	assert(a.rows == b.rows and a.cols == b.cols)

	var result: Matrix = Matrix.new(a.rows, b.cols)
	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = min(a.data[row][col], b.data[row][col])

	return result

static func norm(matrix: Matrix) -> float:
	var sum_of_squares: float = 0.0

	for row in range(matrix.rows):
		for col in range(matrix.cols):
			sum_of_squares += pow(matrix.data[row][col], 2)

	return sqrt(sum_of_squares)


static func square(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = pow(matrix.data[row][col], 2)

	return result

static func power(matrix: Matrix, _power: float) -> Matrix:
	var result: Matrix = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = pow(matrix.data[row][col], _power)

	return result

static func scalar_denominator(scalar: float, matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = scalar / matrix.data[row][col]

	return result

static func square_root(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new(matrix.rows, matrix.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = sqrt(matrix.data[row][col])

	return result

# preferable for ReLU type activation functions
static func uniform_he_init(matrix: Matrix, input_nodes: int) -> Matrix:
	var result: Matrix = Matrix.new(matrix.rows, matrix.cols)

	var limit: float = sqrt(6.0 / float(input_nodes))

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = randf_range(-limit, limit)

	return result

# Preferable for tanh or sigmoid type activation functions
static func uniform_glorot_init(matrix: Matrix, input_nodes: int, output_nodes: int) -> Matrix:
	var result: Matrix = Matrix.new(matrix.rows, matrix.cols)

	# Calculate the range limit for Glorot initialization
	var limit = sqrt(6.0 / float(input_nodes + output_nodes))

	# Fill the matrix with random values within the range [-limit, limit]
	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = randf_range(-limit, limit)

	return result


static func scalar_add(matrix: Matrix, scalar: float) -> Matrix:
	var result = Matrix.new(matrix.rows, matrix.cols)
	for row in range(matrix.rows):
		for col in range(matrix.cols):
			result.data[row][col] = matrix.data[row][col] + scalar
	return result

static func dot_divide(a: Matrix, b: Matrix) -> Matrix:
	assert(a.cols == b.rows)

	var result = Matrix.new(a.rows, b.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = 0.0
			for k in range(a.cols):
				result.data[row][col] += b.data[k][col] / a.data[row][k]

	return result

static func divide(a: Matrix, b: Matrix) -> Matrix:
	assert(a.rows == b.rows and a.cols == b.cols)

	var result: Matrix = Matrix.new(a.rows, a.cols)

	for row in range(result.rows):
		for col in range(result.cols):
			result.data[row][col] = a.data[row][col] / b.data[row][col]

	return result

static func outer_product(a: Matrix, b: Matrix) -> Matrix:
	assert(a.cols == 1 and b.cols == 1)  # Ensure both are column vectors

	var result = Matrix.new(a.rows, b.rows)

	for i in range(a.rows):
		for j in range(b.rows):
			result.data[i][j] = a.data[i][0] * b.data[j][0]

	return result

static func sum(matrix: Matrix) -> float:
	var sum_value: float = 0.0
	for row in range(matrix.rows):
		for col in range(matrix.cols):
			sum_value += matrix.data[row][col]
	return sum_value

static func max(matrix: Matrix) -> float:
	var max_value: float = matrix.data[0][0]
	for row in range(matrix.rows):
		for col in range(matrix.cols):
			max_value = max(max_value, matrix.data[row][col])
	return max_value
