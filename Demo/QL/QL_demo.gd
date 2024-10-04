class_name Minimax

var result_func: Callable
var terminal_func: Callable
var utility_func: Callable
var possible_actions_func: Callable

var is_adversary: bool = false
var states_explored: int = 0
var depth_reached: int = 0
var max_depth: int

func _init(result_func: Callable, terminal_func: Callable, utility_func: Callable, possible_actions_func: Callable, depth: int = -1):
	self.result_func = result_func
	self.terminal_func = terminal_func
	self.utility_func = utility_func
	self.possible_actions_func = possible_actions_func
	self.max_depth = depth

# Modified action function with alpha-beta pruning.
func action(state: Array, depth: int = -1) -> Array:
	var possible_actions: Array = possible_actions_func.call(state)
	var optimal_action: Array
	var optimal_value: float = -INF if not is_adversary else INF
	
	depth_reached += 1

	# Initialize alpha and beta for pruning.
	var alpha: float = -INF
	var beta: float = INF

	# Iterate through all possible actions.
	for _action in possible_actions:
		var result_state = result_func.call(state, _action, not is_adversary)
		# Call minimax with alpha and beta values.
		var value_of_result_state: float = self.minimax(result_state, not is_adversary, alpha, beta)
		match not is_adversary:
			true:
				if value_of_result_state > optimal_value:
					optimal_action = _action
					optimal_value = value_of_result_state
				# Update alpha.
				alpha = max(alpha, optimal_value)
			false:
				if value_of_result_state < optimal_value:
					optimal_action = _action
					optimal_value = value_of_result_state
				# Update beta.
				beta = min(beta, optimal_value)

		# Prune if alpha is greater or equal to beta.
		if beta <= alpha:
			break

	return optimal_action

# Modified minimax function with alpha-beta pruning.
func minimax(state: Array, _is_adversary: bool, alpha: float, beta: float) -> float:
	# Base case: Check if the state is terminal or maximum depth reached.
	if terminal_func.call(state) == true or (max_depth != -1 and depth_reached >= max_depth):
		depth_reached = 0
		return utility_func.call(state, not _is_adversary)

	var possible_actions: Array = possible_actions_func.call(state)
	var optimal_value: float = -INF if not _is_adversary else INF

	depth_reached += 1

	# Iterate through possible actions.
	for _action in possible_actions:
		var result_state = result_func.call(state, _action, _is_adversary)
		states_explored += 1
		var value_of_result_state: float = self.minimax(result_state, not _is_adversary, alpha, beta)

		match not _is_adversary:
			# Maximize the value for non-adversary (player).
			true:
				if value_of_result_state > optimal_value:
					optimal_value = value_of_result_state
				# Update alpha value.
				alpha = max(alpha, optimal_value)
			# Minimize the value for adversary.
			false:
				if value_of_result_state < optimal_value:
					optimal_value = value_of_result_state
				# Update beta value.
				beta = min(beta, optimal_value)

		# Prune the branches if alpha is greater or equal to beta.
		if beta <= alpha:
			break

	return optimal_value
