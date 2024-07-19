class_name Minimax

var result_func: Callable
var terminal_func: Callable
var utility_func: Callable
var possible_actions_func: Callable

var is_adversary: bool = false
var states_explored: int = 0

func _init(result_func: Callable, terminal_func: Callable, utility_func: Callable, possible_actions_func: Callable):
	self.result_func = result_func
	self.terminal_func = terminal_func
	self.utility_func = utility_func
	self.possible_actions_func = possible_actions_func


func action(state: Array) -> Array:
	
	var possible_actions: Array = possible_actions_func.call(state)
	var optimal_action: Array
	var optimal_value: float = -INF if not is_adversary else INF
	
	for _action in possible_actions:
		var result_state = result_func.call(state, _action, not is_adversary)
		var value_of_result_state: float = self.minimax(result_state, not is_adversary)
		match not is_adversary:
			true:
				if value_of_result_state > optimal_value:
					optimal_action = _action
					optimal_value = value_of_result_state
			false:
				if value_of_result_state < optimal_value:
					optimal_action = _action
					optimal_value = value_of_result_state
	
	return optimal_action


func minimax(state: Array, _is_adversary: bool) -> float:
	if terminal_func.call(state) == true:
		return utility_func.call(state, not _is_adversary)
	
	var possible_actions: Array = possible_actions_func.call(state)
	var optimal_value: float = -INF if not _is_adversary else INF
	
	for _action in possible_actions:
		var result_state = result_func.call(state, _action, _is_adversary)
		states_explored += 1
		var value_of_result_state: float = self.minimax(result_state, not _is_adversary)
		match not _is_adversary:
			true:
				if value_of_result_state > optimal_value:
					optimal_value = value_of_result_state
			false:
				if value_of_result_state < optimal_value:
					optimal_value = value_of_result_state
	
	return optimal_value
