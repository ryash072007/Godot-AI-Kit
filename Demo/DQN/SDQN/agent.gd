extends CharacterBody2D

# Enumerations for the types of objects and possible actions
enum objects {NONE, BAD, GOOD}
enum actions {UP, DOWN, LEFT, RIGHT}

@export var epoch_reset: int = 16

# Character movement speed and maximum raycast sensing distance
const speed: int = 200
@onready var MAX_DISTANCE: float = 250 # Maximum distance for raycasts to detect objects
@export var check_fitness: bool = false

# Initialize the Deep Q-Network (DQN) with 24 state inputs and 4 possible actions
var DQN: SDQN = SDQN.new(24, 4)
var prev_state: Array = [] # Previous state of the environment
var prev_action: int = -1 # Previous action taken by the agent
var reward: float = 0 # Current reward for the agent
var done: bool = false # Whether the episode is over
var total_reward: float = 0 # Cumulative reward for the current episode
var total_reward_epoch: float = 0
var resets: int = -1 # Number of times the environment has been reset
var epoch: int = 0
#var max_length_on_screen: float = 1321.0

@export var debug: bool = false


var best_dqn: SDQN
var best_avg_epoch_reward: float = -INF


var prev_EP: float = 0.0

func _ready() -> void:
	$ColorRect.color = Color(randf(), randf(), randf())
	DQN.automatic_decay = true # Disable automatic decay of exploration probability
	reset()


# Function to calculate distance and object type detected by the raycast
func get_distance_and_object(_raycast: RayCast2D) -> Array:
	var colliding: float = 0.0 # Default value if no collision detected
	var distance: float = 0.0
	var object: int = objects.NONE # Default object type is NONE
	if _raycast.is_colliding(): # If the raycast collides with an object
		colliding = 1.0
		var origin: Vector2 = _raycast.global_transform.get_origin() # Origin of the raycast
		var collision: Vector2 = _raycast.get_collision_point()
		object = _raycast.get_collider().get_groups()[0].to_int()
		distance = origin.distance_to(collision) / MAX_DISTANCE
	return [colliding, distance, object] # Return distance and object type

# Function to get the current state for the agent
func get_state() -> Array:
	var state: Array = []
	for raycast in $raycasts.get_children(): # Iterate through all raycasts
		state.append_array(get_distance_and_object(raycast)) # Append the raycast information to the state
	return state

# Function to reset the environment after an episode ends
func reset():
	resets += 1
	total_reward_epoch += total_reward

	if debug:
		print("***************************")
		print("Epoch: " + str(epoch))
		print("Total resets this epoch: " + str(resets))
		print("exploration_probability: " + str(DQN.exploration_probability))
		print("total_reward this reset: " + str(total_reward))
		print("average reward this epoch: " + str(total_reward_epoch / resets))
		print("Current learning rate: " + str(DQN.learning_rate))

	# Reset important variables
	reward = 0
	total_reward = 0
	prev_state = []
	prev_action = -1
	done = false


	if resets % epoch_reset == 0:
		#print("********- " + str($ColorRect.color) + " -********")
		#print("Epoch: " + str(epoch))
		#print("exploration_probability: " + str(DQN.exploration_probability))
		#print("average reward this epoch: " + str(total_reward_epoch / resets))
		#print("Current learning rate: " + str(DQN.learning_rate))

		#DQN.exploration_probability = max(DQN.min_exploration_probability, DQN.exploration_probability - DQN.exploration_decay)
		epoch += 1
		resets = 0
		total_reward_epoch = 0

	# Randomly reposition the agent on the map
	global_position = Vector2(randi_range(40, 1150), randi_range(40, 600))
	$max_life.start() # Start the timer for the episode

# Main loop of the game, called every frame
func _process(delta: float) -> void:
	 #For testing: manually adjust exploration probability using keyboard input
	if Input.is_action_just_pressed("ui_up"):
		prev_EP = DQN.exploration_probability
		DQN.exploration_probability = 0.01
	if Input.is_action_just_pressed("ui_down"):
		DQN.exploration_probability = prev_EP


	# Get the current state
	var current_state: Array = get_state()

	if not done and randf() < 0.3:
		DQN.add_memory(prev_state, prev_action, reward, current_state)

	#DQN.add_memory(prev_state, prev_action, reward, current_state)

	if done == true:
		DQN.add_memory(prev_state, prev_action, reward, current_state)
		reset()

	total_reward += reward
	reward = 0 # Reset reward after applying it
	# Choose an action using the DQN
	var current_action: int = DQN.choose_action(current_state)


	 #Small demerit for each step
	reward -= 0.01


	# Move the agent based on the chosen action
	match current_action:
		actions.UP: position.y -= speed * delta
		actions.DOWN: position.y += speed * delta
		actions.LEFT: position.x -= speed * delta
		actions.RIGHT: position.x += speed * delta

	# Update previous state and action
	prev_state = current_state
	prev_action = current_action


# Called when the timer for the episode runs out
func _on_max_life_timeout() -> void:
	done = true # End the episode


func _on_obj_dec_area_entered(area: Area2D) -> void:
	if area.is_in_group("1"):
		reward -= 10
		done = true
	elif area.is_in_group("2"):
		reward += 50
		done = true
