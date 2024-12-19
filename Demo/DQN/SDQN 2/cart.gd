extends RigidBody2D

# Force applied to move the cart
@export var force_magnitude: float = 1000.0

# Initial x-position so that the position of object doesnt change state
@onready var initial_cart_position: float = global_position.x

# Toggle to test movement
@export var human_testing: bool = false

# Action or input to control the cart (2 for left, 1 for right, 0 for none)
var action: int = 0

var reset: int = -1

# Enviroment Reset Variables
var max_angle: float = deg_to_rad(45.0)

# DQN variables
var DQN: SDQN = SDQN.new(4, 3)
var prev_state: Array
var prev_action: int
var reward: float = 0.0
var done: bool = false
var done_last_frame: bool = false

# Add memory variable
var total_reward: float = 0.0

func _ready() -> void:
	DQN.automatic_decay = true
	reset_environment()

func _physics_process(_delta):

	var direction: int

	# For human testing
	if human_testing:
		if Input.is_action_pressed("ui_left"):
			direction = -1
			action = 2
		elif Input.is_action_pressed("ui_right"):
			direction = 1
			action = 1
		else:
			direction = 0
			action = 0
	else:
		var state: Array = get_state()
		direction = DQN.predict(state)

		if Input.is_action_pressed("ui_accept"):
			done_last_frame = true

		# DQN adding memory and recycling state and action if not done last step
		if done_last_frame:
			done_last_frame = false
		else:
			var reward: float = get_reward()
			total_reward += reward
			DQN.add_memory(prev_state, prev_action, reward, state)
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
	if absf($pole.rotation) > max_angle:
		done = true
		return -100.0
	else:
		return 1


func reset_environment() -> void:

	reset += 1

	print("___________________________________")
	print("Reset: " + str(reset))
	print("Learning Rate: " + str(DQN.learning_rate))
	print("Exploration Rate: " + str(DQN.exploration_probability))
	print("Total reward: " + str(total_reward))

	done = false
	done_last_frame = true
	total_reward = 0

	global_position.x = initial_cart_position
	linear_velocity = Vector2.ZERO
	angular_velocity = 0

	$pole.rotation = randf_range(-0.05, 0.05)
	$pole.angular_velocity = 0


# Incase get_reward misses it
func _on_pole_body_entered(body: Node) -> void:
	done = true


func _on_existence_timeout() -> void:
	done = true
	print("Timed Out")
	$existence.start()
