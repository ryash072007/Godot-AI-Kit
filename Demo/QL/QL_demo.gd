extends Node2D

var qnet: QLearning
var row: int = 0
var column: int = 0
var reward_states = [4, 24, 35]
var punish_states = [3, 18, 21, 28, 31]
var current_state: int = 0
var previous_reward: float = 0.0
var done: bool = false

func _ready() -> void:
	qnet = QLearning.new(36, 4)
	qnet.decay_per_steps = 75

func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		$control.wait_time = 0.5
	elif Input.is_action_just_pressed("ui_down"):
		$control.wait_time = 0.05

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

func reset() -> void:
	row = 0
	column = 0
	done = false
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))

func _on_control_timeout() -> void:
	if done:
		reset()
		return
	current_state = row * 6 + column
	var action_to_do: int = qnet.predict(current_state, previous_reward)
	previous_reward = 0.0
	if is_out_bound(action_to_do):
		previous_reward -= 0.75
		done = true
	elif row * 6 + column in punish_states:
		previous_reward -= 0.5
		done = true
	elif row * 6 + column in reward_states:
		previous_reward += 1.0
		done = true
	else:
		previous_reward -= 0.05
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))
	$lr.text = str(qnet.exploration_probability)
