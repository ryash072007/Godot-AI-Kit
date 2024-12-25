extends RigidBody2D

# Configuration for cart movement and physics
@export var force_magnitude: float = 1000.0

# Initial x-position so that the position of object doesnt change state
@onready var initial_cart_position: float = global_position.x

# Toggle to test movement
@export var human_testing: bool = false

# Neural Network configuration
@export_enum("SGD", "ADAM") var optimiser: int

@export_range(0.0001, 0.1, 0.0001) var learning_rate: float = 0.0001

# Cart control variables
# 2 = no movement, 1 = right, 0 = left
var action: int = 0

var reset: int = -1

# Failure conditions for the simulation
var max_angle: float = deg_to_rad(25.0) # Cart fails if pole tilts beyond this angle
var threshold_distace: float = 250.0 # Cart fails if it moves too far from start

# Deep Q-Network (DQN) setup
# 4 inputs (state variables), 2 outputs (actions)
var DQN: SDQN = SDQN.new(4, 2)
var prev_state: Array
var prev_action: int
var reward: float = 0.0
var done: bool = false
var done_last_frame: bool = false

# Add memory variable
var total_reward: float = 0.0 # Tracks cumulative reward for current episode

# File handling for logging and model saving

@export var log_data: bool = true

@export var log_file_name: String

@export var SDQN_file_name: String

@export var enabled: bool = true

@export var is_learning: bool = true

var log_file: FileAccess

func _ready() -> void:
	Engine.max_fps = 24

	if not enabled:
		get_parent().remove_child.call_deferred(get_node("."))

	if log_data:
		log_file = FileAccess.open("user://" + log_file_name, FileAccess.WRITE)
		log_file.store_string("Reset, Exploration Probability, Total Reward, Time Alive\n")


	#var Q_network: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new(optimiser)
	#Q_network.add_layer(4)
	#Q_network.add_layer(16, "ELU")
	#Q_network.add_layer(16, "ELU")
	#Q_network.add_layer(2, "LINEAR")
	#Q_network.learning_rate = learning_rate
	#DQN.set_Q_network(Q_network)
	#DQN.set_clip_value(10)
	#DQN.automatic_decay = true
	#DQN.set_lr_value(learning_rate)
	DQN.load("user://ADAM_001_ELU - Copy (3).ryash")


	$sprite.color = Color(randf(), randf(), randf())
	$pole/sprite.color = $sprite.color


	reset_environment()

func _physics_process(_delta: float) -> void:
	# The function handles both human testing and AI control modes
	# Updates DQN memory and applies forces to the cart

	if not is_learning:
		done_last_frame = true

	var direction: int

	# For human testing
	if human_testing:
		if Input.is_action_pressed("ui_left"):
			direction = -1
			action = 0
		elif Input.is_action_pressed("ui_right"):
			direction = 1
			action = 1
		else:
			direction = 0
			action = 2
		get_reward()
	else:
		var state: Array = get_state()
		action = DQN.choose_action(state)
		match action:
			0:
				direction = -1
			1:
				direction = 1
			2:
				direction = 0

		if Input.is_action_pressed("ui_accept"):
			done_last_frame = true

		# DQN adding memory and recycling state and action if not done last step
		if done_last_frame:
			total_reward += get_reward()
			done_last_frame = false
		else:
			reward = get_reward()
			total_reward += reward
			DQN.add_memory(prev_state, prev_action, reward, state, done)
		prev_state = state
		prev_action = action

	# Apply force to move the cart if not done
	if done:
		reset_environment()
	else:
		apply_force(Vector2(force_magnitude * direction, 0))


func get_state() -> Array:
	# Returns current state of the system as an array:
	# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

	var cart_position: float = global_position.x - initial_cart_position
	var cart_velocity: float = linear_velocity.x
	var pole_angle: float = $pole.rotation
	var pole_angular_velocity: float = $pole.angular_velocity

	var state: Array = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

	return state


func get_reward() -> float:
	# Returns -100 if cart fails (pole falls below a set angle or cart moves too far)
	# Returns 1.0 for each successful step

	if absf($pole.rotation) > max_angle or abs(global_position.x) - initial_cart_position > threshold_distace:
		done = true
		return -100.0
	else:
		return 1.0


func reset_environment() -> void:
	# Resets the simulation:
	# 1. Saves model every 16 resets
	# 2. Updates display information
	# 3. Logs performance metrics
	# 4. Resets cart and pole to initial positions with slight randomization

	reset += 1

	if log_data:
		if reset % 16:
			DQN.save("user://" + SDQN_file_name)

	#print("_______________ " + str($sprite.color) + " _______________")
	var info: String = "Reset: " + str(reset) + "\nLearning Rate: " + str(DQN.learning_rate) + "\nExploration Rate: " + str(DQN.exploration_probability) + "\nTotal reward: " + str(total_reward) + "\nTime Alive: " + str(5 - $existence.time_left)
	#print(info)
	$info.text = info

	if log_data:
		log_file.store_string(str(reset) + ', ' + str(DQN.exploration_probability) + ", " + str(total_reward) + ", " + str(5 - $existence.time_left) + '\n')
		log_file.flush()

	done = false
	done_last_frame = true
	total_reward = 0
	$existence.stop()
	$existence.start()

	global_position.x = initial_cart_position
	linear_velocity = Vector2.ZERO
	angular_velocity = 0

	randomize()
	$pole.rotation = randf_range(-0.0075, 0.0075)
	$pole.rotation = 0
	$pole.angular_velocity = 0
	$pole.linear_velocity = Vector2.ZERO


# Event handlers
func _on_pole_body_entered(_body: Node) -> void:
	# Fail-safe to catch pole collisions
	# Incase get_reward misses it

	done = true


func _on_existence_timeout() -> void:
	# Handles timeout of current episode

	reset_environment()
	print("Timed Out")


func _on_tree_exiting() -> void:
	# Cleanup when scene exits

	log_file.close()
	DQN.close_threading()
