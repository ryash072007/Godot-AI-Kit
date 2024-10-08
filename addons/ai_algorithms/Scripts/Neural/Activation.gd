class_name Activation

static func sigmoid(value: float, _row: int, _col: int) -> float:
	return 1 / (1 + exp(-value))

static func dsigmoid(value: float, _row: int, _col: int) -> float:
	return value * (1 - value)

static func relu(value: float, _row: int, _col: int) -> float:
	return max(0.0, value)

static func drelu(value: float, _row: int, _col: int) -> float:
	return 1.0 if value > 0 else 0.0

static func tanh_(value: float, _row: int, _col: int) -> float:
	return tanh(value)

static func dtanh(value: float, _row: int, _col: int) -> float:
	return 1 - pow(tanh(value), 2)

static func arcTan(value: float, _row: int, _col: int) -> float:
	return pow(tan(value), -1)

static func darcTan(value: float, _row: int, _col: int) -> float:
	return 1 / (pow(value, 2) + 1)

static func prelu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	return (alpha * value) if value < 0 else value

static func dprelu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	return alpha if value < 0 else 1

static func elu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	if value < 0:
		return alpha * (exp(value) - 1)
	else:
		return value

static func delu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	return (((alpha * (exp(value) - 1)) if value < 0 else value) + alpha) if value < 0 else 1

static func softplus(value: float, _row: int, _col: int) -> float:
	return log(exp(1)) * (1 + exp(value))

static func dsoftplus(value: float, _row: int, _col: int) -> float:
	return 1 / (1 + exp(-value))

static func linear(value: float, _row: int, _col: int) -> float:
	return value

static func dlinear(value: float, _row: int, _col: int) -> float:
	return 1
