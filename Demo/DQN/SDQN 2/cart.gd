extends RigidBody2D

# Force applied to move the cart
@export var force_magnitude: float = 1000.0

# Initial x-position so that the position of object doesnt change state
@onready var initial_cart_position: float = global_position.x

# Toggle to test movement
@export var human_testing: bool = false

@export_range(0.0001, 0.1, 0.0001) var learning_rate: float = 0.0001

# Action or input to control the cart (2 for none, 1 for right, 0 for left)
var action: int = 0

var reset: int = -1

# Enviroment Reset Variables
var max_angle: float = deg_to_rad(25.0)
var threshold_distace: float = 250.0

# DQN variables
var DQN: SDQN = SDQN.new(4, 2)
var prev_state: Array
var prev_action: int
var reward: float = 0.0
var done: bool = false
var done_last_frame: bool = false

# Add memory variable
var total_reward: float = 0.0

var log_file: FileAccess = FileAccess.open("user://cart_data_SGD_multi.csv", FileAccess.WRITE)

func _ready() -> void:

	Engine.max_fps = 24

	var Q_network: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new(NeuralNetworkAdvanced.methods.SGD)
	Q_network.add_layer(4)
	Q_network.add_layer(16, Q_network.ACTIVATIONS.PRELU)
	Q_network.add_layer(16, Q_network.ACTIVATIONS.PRELU)
	Q_network.add_layer(2, Q_network.ACTIVATIONS.LINEAR)
	Q_network.learning_rate = learning_rate
	DQN.set_Q_network(Q_network)

	$sprite.color = Color(randf(), randf(), randf())
	$pole/sprite.color = $sprite.color
	DQN.automatic_decay = true
	DQN.set_clip_value(100.0)
	DQN.lr_decay_rate = 1
	DQN.set_lr_value(learning_rate)
	DQN.use_threading()

	log_file.store_string("Reset, Exploration Probability, Total Reward, Time Alive\n")

	reset_environment()

func _physics_process(_delta: float) -> void:

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
			done_last_frame = false
		else:
			var reward: float = get_reward()
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
	var cart_position: float = global_position.x - initial_cart_position
	var cart_velocity: float = linear_velocity.x
	var pole_angle: float = $pole.rotation
	var pole_angular_velocity: float = $pole.angular_velocity

	var state: Array = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

	return state


func get_reward() -> float:
	if absf($pole.rotation) > max_angle or abs(global_position.x) - initial_cart_position > threshold_distace:
		done = true
		return -100.0
	else:
		return 1.0


func reset_environment() -> void:

	reset += 1

	#print("_______________ " + str($sprite.color) + " _______________")
	var info: String = "Reset: " + str(reset) + "\nLearning Rate: " + str(DQN.learning_rate) + "\nExploration Rate: " + str(DQN.exploration_probability) + "\nTotal reward: " + str(total_reward) + "\nTime Alive: " + str(20 - $existence.time_left)
	#print(info)
	$info.text = info

	log_file.store_string(str(reset) + ', ' + str(DQN.exploration_probability) + ", " + str(total_reward) + ", " + str(20 - $existence.time_left) + '\n')
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
	$pole.rotation = randf_range(-0.02, 0.02)
	$pole.rotation = 0
	$pole.angular_velocity = 0
	$pole.linear_velocity = Vector2.ZERO


# Incase get_reward misses it
func _on_pole_body_entered(body: Node) -> void:
	done = true


func _on_existence_timeout() -> void:
	reset_environment()
	print("Timed Out")


func _on_tree_exiting() -> void:
	log_file.close()
	DQN.close_threading()
