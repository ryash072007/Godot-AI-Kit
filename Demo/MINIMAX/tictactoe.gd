extends Node2D

var minimax: Minimax = Minimax.new(Callable(result),
Callable(terminal),
Callable(utility),
Callable(possible_actions)
)

var turn = "X"
var _board = [[null, null, null],[null, null, null],[null, null, null]]

# Called when the node enters the scene tree for the first time.
func _ready():
	minimax.is_adversary = true


func result(board: Array, action: Array, is_adversary: bool) -> Array:
	var __board = board.duplicate(true)
	__board[action[1]][action[0]] = "O" if is_adversary else "X"
	return __board


func terminal(board: Array) -> bool:
	if utility(board, false) != 0:
		return true
	for row in board:
		for cell in row:
			if cell == null:
				return false
	return true


func utility(board: Array, is_adversary: bool) -> float:
	for row in board:
		if row[0] == row[1] and row[1] == row[2] and row[0] != null:
			return 1 if not is_adversary else -1
	
	for col_index in range(3):
		if board[0][col_index] == board[1][col_index] and board[1][col_index] == board[2][col_index] and board[0][col_index] != null:
			return 1 if not is_adversary else -1
	
	if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != null:
		return 1 if not is_adversary else -1
	
	if board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] != null:
		return 1 if not is_adversary else -1
	
	return 0

func possible_actions(board: Array) -> Array[Array]:
	var _possible_action: Array[Array] = []
	for row_index in range(3):
		var row = board[row_index]
		for col_index in range(3):
			var cell_data = row[col_index]
			if cell_data == null:
				_possible_action.append([col_index, row_index])
	return _possible_action


func change_turn():
	match turn:
		"X":
			turn = "O"
		"O":
			turn = "X"


func grid_cell_clicked(pos: Vector2i) -> void:
	if _board[pos.y][pos.x] != null:
		return
	_board[pos.y][pos.x] = turn
	var grid_cell = get_node("board/%s/%s" % [pos.y, pos.x])
	grid_cell.text = turn
	
	change_turn()
	
	var action_to_do: Array = minimax.action(_board)
	if not action_to_do:
		return
	_board[action_to_do[1]][action_to_do[0]] = turn
	grid_cell = get_node("board/%s/%s" % [action_to_do[1], action_to_do[0]])
	grid_cell.text = turn
	change_turn()
