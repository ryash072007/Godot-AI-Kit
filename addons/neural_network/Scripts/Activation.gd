class_name Activation

static func sigmoid(value: float, _row: int, _col: int) -> float:
	return 1 / (1 + exp(-value))

static func dsigmoid(value: float, _row: int, _col: int) -> float:
	return value * (1 - value)
