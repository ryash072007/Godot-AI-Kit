extends CharacterBody2D

enum objects {NONE, BAD, GOOD}
enum actions {UP, DOWN, LEFT, RIGHT}

const speed: int = 100
@onready var MAX_DISTANCE: float = 150
@onready var goodObjPos: Vector2 = $"../Map/good/GOOD".global_position

var DQN: SDQN = SDQN.new(19, 4)
var prev_state: Array = []
var prev_action: int = -1
var reward: float = 0
var done: bool = false

var total_reward: float = 0

var resets: int = 0

@onready var prev_distance_to_goal: float = global_position.distance_to(goodObjPos)

func _ready() -> void:
	DQN.automatic_decay = false

func get_distance_and_object(_raycast: RayCast2D) -> Array:
	var distance: float = -1
	var object: int = objects.NONE
	if _raycast.is_colliding():
		var origin: Vector2 = _raycast.global_transform.get_origin()
		var collision: Vector2 = _raycast.get_collision_point()
		object = _raycast.get_collider().get_groups()[0].to_int()
		distance = origin.distance_to(collision) / MAX_DISTANCE
	return [distance, object]

func get_state() -> Array:
	var state: Array = []
	for raycast in $raycasts.get_children():
		var data: Array = get_distance_and_object(raycast)
		state.append_array(data)
	# Normalized distance to goal
	#state.append(global_position.distance_to(goodObjPos) / MAX_DISTANCE)
	state.append(global_position.distance_to($"../Map/good/GOOD".global_position) / MAX_DISTANCE)
	state.append(global_position.distance_to($"../Map/good2/GOOD".global_position) / MAX_DISTANCE)
	state.append(global_position.distance_to($"../Map/good3/GOOD".global_position) / MAX_DISTANCE)
	return state

func _process(delta: float) -> void:

	if Input.is_action_just_pressed("ui_up"):
		DQN.exploration_probability = 0.01
	if Input.is_action_just_pressed("ui_down"):
		DQN.exploration_probability = 0.6

	var current_state: Array = get_state()
	var current_action: int = DQN.choose_action(current_state)

	if prev_action != -1:
		if done == false:
			#var goodDistance: float = global_position.distance_to(goodObjPos)

			## Reward shaping
			#if goodDistance < prev_distance_to_goal:
				#reward += 0.1  # Increased reward for moving closer
			#else:
				#reward -= 0.2  # Larger penalty for moving further away

			# Penalize each step slightly to encourage faster decisions
			reward -= 0.01

			#prev_distance_to_goal = goodDistance
		total_reward += reward
		DQN.add_memory(prev_state, prev_action, reward, current_state)

	if done == true:
		reset()

	match current_action:
		actions.UP:
			position.y -= speed * delta
		actions.DOWN:
			position.y += speed * delta
		actions.LEFT:
			position.x -= speed * delta
		actions.RIGHT:
			position.x += speed * delta

	prev_state = current_state
	prev_action = current_action

	reward = 0

func _on_bad_body_entered(_body: Node2D) -> void:
	reward -= 1  # Penalty for hitting bad object
	done = true

func reset():
	print("***************************")
	print("exploration_probability: " + str(DQN.exploration_probability))
	print("total_reward: " + str(total_reward))

	reward = 0
	total_reward = 0
	prev_state = []
	prev_action = -1
	done = false
	resets += 1
	prev_distance_to_goal = INF

	# Gradual decay of exploration probability
	if resets >= 4:
		resets = 0
		DQN.exploration_probability = max(DQN.min_exploration_probability, DQN.exploration_probability - DQN.exploration_decay)

	global_position = Vector2(randi_range(50, 1100), randi_range(50, 590))
	$max_life.start()

func _on_boundary_body_entered(body: Node2D) -> void:
	reward -= 1  # Penalty for hitting the boundary
	done = true

func _on_good_body_entered(body: Node2D) -> void:
	reward += 10  # Larger reward for reaching the goal
	done = true

func _on_max_life_timeout() -> void:
	reward -= 0.1
	done = true
