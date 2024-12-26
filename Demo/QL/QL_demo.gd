extends Node2D

# Neural network for Q-Learning and game state variables
var qnet: QLearning
var row: int = 0
var column: int = 0
# States that give positive rewards (goal states)
var reward_states = [4, 24, 35]
# States that give negative rewards (obstacles)
var punish_states = [3, 18, 21, 28, 31]
var current_state: int = 0
var previous_reward: float = 0.0
var done: bool = false

# Initialize Q-Learning network with 36 states (6x6 grid) and 4 possible actions (movement directions)
func _ready() -> void:
	Engine.max_fps = 24
	qnet = QLearning.new(36, 4)
	qnet.decay_per_steps = 75

# Handle input for controlling simulation speed
func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		$control.wait_time = 0.2
	elif Input.is_action_just_pressed("ui_down"):
		$control.wait_time = 0.05

# Check if the chosen action would move the agent outside the grid
# Actions: 0=left, 1=down, 2=right, 3=up
func is_out_bound(action: int) -> bool:
	var _column := column
	var _row := row
	match action:
		0: _column -= 1
		1: _row += 1
		2: _column += 1
		3: _row -= 1
	if _column < 0 or _row < 0 or _column > 5 or _row > 5:
		return true
	else:
		column = _column
		row = _row
		return false

# Reset agent to starting position
func reset() -> void:
	$player.color = Color(randf_range(0.3, 1), randf_range(0.3, 1), randf_range(0.3, 1))
	row = randi_range(1, 2)
	column = randi_range(0, 5)
	done = false
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))

# Main game loop - called on timer timeout
func _on_control_timeout() -> void:
	if done:
		reset()
		return

	# Calculate current state from grid position
	current_state = row * 6 + column
	# Get next action from Q-Learning network
	var action_to_do: int = qnet.predict(current_state, previous_reward)
	previous_reward = 0.0


	# Apply rewards/punishments based on action results
	if is_out_bound(action_to_do):
		previous_reward -= 0.75  # Penalize for hitting boundaries
		done = true
	elif row * 6 + column in punish_states:
		previous_reward -= 0.5   # Penalize for hitting obstacles
		done = true
	elif row * 6 + column in reward_states:
		previous_reward += 1.0   # Reward for reaching goals
		done = true
	else:
		previous_reward -= 0.05  # Small penalty for each move to encourage efficiency

	var tween := create_tween()
	tween.tween_property($player, "position", Vector2(96 * column + 16, 512 - (96 * row + 16)), 0.05)
	# Update agent position on screen
	#$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))
	$lr.text = str(qnet.exploration_probability)
