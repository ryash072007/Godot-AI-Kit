extends Node2D

# Initialize the Minimax AI with required callback functions
var minimax: Minimax = Minimax.new(Callable(result),
Callable(terminal),
Callable(utility),
Callable(possible_actions)
)

# Game state variables
var turn = "X"  # Current player's turn
var _board = [[null, null, null], [null, null, null], [null, null, null]]  # 3x3 game board
var _is_adversary: bool = true  # Whether AI plays as O (true) or X (false)

# Initialize the game
func _ready():
	if _is_adversary:
		minimax.is_adversary = _is_adversary
	else:
		ai_minimax()

# Simulate the result of an action on the board
# Returns a new board state after applying the move
func result(board: Array, action: Array, is_adversary: bool) -> Array:
	var __board = board.duplicate(true)
	__board[action[1]][action[0]] = "O" if is_adversary else "X"
	return __board

# Check if the game has reached a terminal state (win/draw)
func terminal(board: Array) -> bool:
	if utility(board, false) != 0:
		return true
	for row in board:
		for cell in row:
			if cell == null:
				return false
	return true

# Evaluate the board state
# Returns: 1 for win, -1 for loss, 0 for ongoing/draw
func utility(board: Array, is_adversary: bool) -> float:
	# Check rows for win
	for row in board:
		if row[0] == row[1] and row[1] == row[2] and row[0] != null:
			return 1 if not is_adversary else -1
	
	# Check columns for win
	for col_index in range(3):
		if board[0][col_index] == board[1][col_index] and board[1][col_index] == board[2][col_index] and board[0][col_index] != null:
			return 1 if not is_adversary else -1
	
	# Check diagonal (top-left to bottom-right)
	if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != null:
		return 1 if not is_adversary else -1
	
	# Check diagonal (top-right to bottom-left)
	if board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] != null:
		return 1 if not is_adversary else -1
	
	return 0

# Get all possible valid moves on the current board
func possible_actions(board: Array) -> Array[Array]:
	var _possible_action: Array[Array] = []
	for row_index in range(3):
		var row = board[row_index]
		for col_index in range(3):
			var cell_data = row[col_index]
			if cell_data == null:
				_possible_action.append([col_index, row_index])
	return _possible_action

# Switch turns between X and O
func change_turn():
	match turn:
		"X":
			turn = "O"
		"O":
			turn = "X"

# Handle player's move when a cell is clicked
func grid_cell_clicked(pos: Vector2i) -> void:
	# Validate and process player's move
	if _board[pos.y][pos.x] != null:
		return
	_board[pos.y][pos.x] = turn
	var grid_cell = get_node("board/%s/%s" % [pos.y, pos.x])
	grid_cell.text = turn
	change_turn()
	check_game_over()
	
	ai_minimax()

# Check if the game has ended (win/draw)
func check_game_over():
	if utility(_board, _is_adversary) != 0:
		$caption.text = turn + " won! States explored: " + str(minimax.states_explored)

	elif terminal(_board):
		$caption.text = "Draw! States explored: " + str(minimax.states_explored)

# Execute AI's move using minimax algorithm
func ai_minimax() -> void:
	var action_to_do: Array = minimax.action(_board)
	if not action_to_do:
		return
	_board[action_to_do[1]][action_to_do[0]] = turn
	var grid_cell = get_node("board/%s/%s" % [action_to_do[1], action_to_do[0]])
	grid_cell.text = turn
	change_turn()
	check_game_over()
