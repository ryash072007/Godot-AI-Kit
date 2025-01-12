class_name Augmentations

static func rotate_random(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new(matrix.cols, matrix.rows)
	var angle: float = randf_range(0, 360)
	angle = deg_to_rad(angle)
	var cos_angle: float = cos(angle)
	var sin_angle: float = sin(angle)

	var cx: float = (matrix.cols - 1) / 2.0
	var cy: float = (matrix.rows - 1) / 2.0
	var result_cx: float = (result.cols - 1) / 2.0
	var result_cy: float = (result.rows - 1) / 2.0

	for row in range(matrix.rows):
		for col in range(matrix.cols):
			var translated_x: float = col - cx
			var translated_y: float = row - cy
			
			var rotated_x: float = translated_x * cos_angle - translated_y * sin_angle
			var rotated_y: float = translated_x * sin_angle + translated_y * cos_angle
			
			var new_col: int = int(round(rotated_x + result_cx))
			var new_row: int = int(round(rotated_y + result_cy))
			
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


