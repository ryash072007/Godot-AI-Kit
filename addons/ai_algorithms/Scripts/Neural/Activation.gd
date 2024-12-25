class_name Activation

class SIGMOID:
	static func function(value: float, _row: int, _col: int) -> float:
		return 1 / (1 + exp(-value))

	static func derivaive(value: float, _row: int, _col: int) -> float:
		return value * (1 - value)

class RELU:
	static func function(value: float, _row: int, _col: int) -> float:
		return max(0.0, value)

	static func derivative(value: float, _row: int, _col: int) -> float:
		return 1.0 if value > 0 else 0.0

class TANH:
	static func function(value: float, _row: int, _col: int) -> float:
		return tanh(value)

	static func derivative(value: float, _row: int, _col: int) -> float:
		return 1 - pow(tanh(value), 2)

class ARCTAN:
	static func function(value: float, _row: int, _col: int) -> float:
		return atan(value)

	static func derivative(value: float, _row: int, _col: int) -> float:
		return 1 / (pow(value, 2) + 1)

class LEAKYRELU:
	static func function(value: float, _row: int, _col: int) -> float:
		var alpha: float = 0.1
		return (alpha * value) if value < 0 else value

	static func derivative(value: float, _row: int, _col: int) -> float:
		var alpha: float = 0.1
		return alpha if value < 0 else 1

class ELU:
	static func function(value: float, _row: int, _col: int) -> float:
		var alpha: float = 0.1
		return (alpha * (exp(value) - 1)) if value < 0 else value

	static func derivative(value: float, _row: int, _col: int) -> float:
		var alpha: float = 0.1
		return alpha * exp(value) if value < 0 else 1

class SOFTPLUS:
	static func function(value: float, _row: int, _col: int) -> float:
		return log(1 + exp(value))

	static func derivative(value: float, _row: int, _col: int) -> float:
		return 1 / (1 + exp(-value))

class LINEAR:
	static func function(value: float, _row: int, _col: int) -> float:
		return value

	static func derivative(value: float, _row: int, _col: int) -> float:
		return 1.0
