class_name Augmentations

static func rotate_random(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new(matrix.cols, matrix.rows)
	var angle = randf_range(0, 360)
	var radians = deg_to_rad(angle)
	var cos_angle = cos(radians)
	var sin_angle = sin(radians)

	for row in range(matrix.rows):
		for col in range(matrix.cols):
			var new_row = int(row * cos_angle - col * sin_angle)
			var new_col = int(row * sin_angle + col * cos_angle)
			if new_row >= 0 and new_row < result.rows and new_col >= 0 and new_col < result.cols:
				result.data[new_row][new_col] = matrix.data[row][col]

	return result

static func flip_horizontal(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.copy(matrix)
	for row in range(matrix.rows):
		result.data[row].reverse()
	return result

static func flip_vertical(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.copy(matrix)
	result.data.reverse()
	return result


