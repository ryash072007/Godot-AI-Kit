extends CharacterBody2D

enum objects {NONE, BAD, GOOD}
enum actions {UP, DOWN, LEFT, RIGHT}

const speed: int = 100
@onready var goodObjPos: Vector2 = $"../Map/good/GOOD".global_position

var DQN: SDQN = SDQN.new(17, 4)
var prev_state: Array = []
var prev_action: int = -1
var reward: float = 0
var done: bool = false

var total_reward: float = 0

@onready var prev_distance_to_goal: float = global_position.distance_squared_to(goodObjPos)

func get_distance_and_object(_raycast: RayCast2D) -> Array:
	var distance: float = 75.0
	var object: int = objects.NONE
	if _raycast.is_colliding():
		var origin: Vector2 = _raycast.global_transform.get_origin()
		var collision: Vector2 = _raycast.get_collision_point()
		object = _raycast.get_collider().get_groups()[0].to_int()
		distance = origin.distance_to(collision)
	return [distance, object]

func get_state() -> Array:
	var state: Array = []
	for raycast in $raycasts.get_children():
		var data: Array = get_distance_and_object(raycast)
		state.append_array(data)
	state.append(global_position.distance_to(goodObjPos))
	return state

func _process(delta: float) -> void:
	var current_state: Array = get_state()
	var current_action: int = DQN.choose_action(current_state)

	if prev_action != -1:
		if done == false:
			var goodDistance: float = global_position.distance_squared_to(goodObjPos)
			if  goodDistance < prev_distance_to_goal:
				reward += 0.05
			else:
				reward -= 0.05
			prev_distance_to_goal = goodDistance
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
	reward -= 0.3
	done = true

func reset():
	print("***************************")
	print("exploration_probability: " + str(DQN.exploration_probability))
	print("total_reward: " + str(total_reward))


	reward = 0
	prev_state = []
	prev_action = -1
	done = false

	global_position = Vector2(randi_range(50, 1100), randi_range(50, 590))
	$max_life.start()



func _on_boundary_body_entered(body: Node2D) -> void:
	reward -= 0.6
	done = true


func _on_good_body_entered(body: Node2D) -> void:
	reward += 0.6
	done = true


func _on_max_life_timeout() -> void:
	reset()
