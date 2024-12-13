extends CharacterBody2D

# Enumerations for the types of objects and possible actions
enum objects {NONE, BAD, GOOD}
enum actions {UP, DOWN, LEFT, RIGHT}

# Character movement speed and maximum raycast sensing distance
const speed: int = 100
@onready var MAX_DISTANCE: float = 150  # Maximum distance for raycasts to detect objects
@onready var goodObjPos: Vector2 = $"../Map/good/GOOD".global_position  # Position of the goal object (good object)

# Initialize the Deep Q-Network (DQN) with 19 state inputs and 4 possible actions
var DQN: SDQN = SDQN.new(18, 4)
var prev_state: Array = []  # Previous state of the environment
var prev_action: int = -1   # Previous action taken by the agent
var reward: float = 0  # Current reward for the agent
var done: bool = false  # Whether the episode is over
var total_reward: float = 0  # Cumulative reward for the current episode
var resets: int = -1  # Number of times the environment has been reset
var epoch: int = 0
@onready var prev_distance_to_goal: float = global_position.distance_to(goodObjPos)  # Previous distance to the goal

var prev_EP: float = 0.0

# Function called when the scene is ready
func _ready() -> void:
	DQN.automatic_decay = false  # Disable automatic decay of exploration probability

# Function to calculate distance and object type detected by the raycast
func get_distance_and_object(_raycast: RayCast2D) -> Array:
	var distance: float = -1  # Default value if no collision detected
	var object: int = objects.NONE  # Default object type is NONE
	if _raycast.is_colliding():  # If the raycast collides with an object
		var origin: Vector2 = _raycast.global_transform.get_origin()  # Origin of the raycast
		var collision: Vector2 = _raycast.get_collision_point()  # Point where the raycast collides
		object = _raycast.get_collider().get_groups()[0].to_int()  # Get the object's type (BAD/GOOD)
		distance = origin.distance_to(collision) / MAX_DISTANCE  # Normalize distance to collision point
	return [distance, object]  # Return distance and object type

# Function to get the current state for the agent
func get_state() -> Array:
	var state: Array = []
	for raycast in $raycasts.get_children():  # Iterate through all raycasts
		state.append_array(get_distance_and_object(raycast))  # Append the raycast information to the state
	state.append(global_position.distance_to(goodObjPos)  / MAX_DISTANCE)  # Add the distance to the goal
	state.append($max_life.time_left)
	print(state)
	return state

# Function to reset the environment after an episode ends
func reset():
	print("***************************")
	print("Epoch: " + str(epoch))
	print("Total resets this epoch: " + str(resets))
	print("exploration_probability: " + str(DQN.exploration_probability))
	print("total_reward: " + str(total_reward))

	# Reset important variables
	reward = 0
	total_reward = 0
	prev_state = []
	prev_action = -1
	done = false
	resets += 1

	# Decay exploration probability gradually every 4 resets
	#if resets % 1 == 0:
		#DQN.exploration_probability = max(DQN.min_exploration_probability, DQN.exploration_probability - DQN.exploration_decay)


	if resets % 8 == 0:
		#var file = FileAccess.open("user://SDQNEpochData.txt", FileAccess.READ_WRITE)
		#file.seek_end()
		#file.store_string("Epoch: " + str(epoch) + " | Total Reward: " + str(total_reward) + " | EP: " + str(DQN.exploration_probability) + '\n')
		##file.store_string("DQN Values: ")
		##file.store_var(DQN.Q_network.network, true)
		##file.store_string("DQN Target Values: ")
		##file.store_var(DQN.target_Q_network.network, true)
		#file.store_string("____________________________________________________")
		#file.close()

		DQN.exploration_probability = max(DQN.min_exploration_probability, DQN.exploration_probability - DQN.exploration_decay)
		epoch += 1
		resets = 0

	# Randomly reposition the agent on the map
	global_position = Vector2(randi_range(700, 1100), randi_range(20, 200))
	prev_distance_to_goal = global_position.distance_to(goodObjPos)  # Reset the distance to goal
	$max_life.start()  # Start the timer for the episode

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
	# Choose an action using the DQN
	var current_action: int = DQN.choose_action(current_state)

	# If there was a previous action, update the DQN with the current reward
	var goodDistance: float = global_position.distance_to(goodObjPos)
	if prev_action != -1:
		if done == false:

			var difference: float = prev_distance_to_goal - goodDistance
			# Reward shaping: Give a small positive reward if getting closer to the goal, negative otherwise
			#reward += 1.5 * difference if difference > 0 else -2.0 * abs(difference)

			if difference > 0:
				reward += 0.01
			else:
				reward -= 0.02

			# Penalize each step slightly to encourage faster decisions
			#reward -= 0.01

		#print(reward)
		total_reward += reward
		DQN.add_memory(prev_state, prev_action, reward, current_state)

	prev_distance_to_goal = goodDistance
	# If the episode is done, reset the environment
	if done:
		print(reward)
		reset()

	# Move the agent based on the chosen action
	match current_action:
		actions.UP: position.y -= speed * delta
		actions.DOWN: position.y += speed * delta
		actions.LEFT: position.x -= speed * delta
		actions.RIGHT: position.x += speed * delta

	# Update previous state and action
	prev_state = current_state
	prev_action = current_action
	reward = 0  # Reset reward after applying it

# Event handlers for different collisions
# Called when the agent hits a bad object
func _on_bad_body_entered(_body: Node2D) -> void:
	reward -= 0.5  # Penalty for hitting a bad object
	done = true  # End the episode

# Called when the agent hits the boundary of the map
func _on_boundary_body_entered(body: Node2D) -> void:
	reward -= 5  # Penalty for hitting the boundary
	done = true  # End the episode

# Called when the agent reaches the goal (good object)
func _on_good_body_entered(body: Node2D) -> void:
	reward += 1  # Large reward for reaching the goal
	done = true  # End the episode

# Called when the timer for the episode runs out
func _on_max_life_timeout() -> void:
	reward -= 0.05  # Small penalty for running out of time
	done = true  # End the episode
