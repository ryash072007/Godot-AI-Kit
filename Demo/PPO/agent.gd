extends Node2D

@onready var ObstaclesAndWalls := $ObstaclesAndWalls
@onready var Goals := $Goals
@onready var parent_node := $".."

func get_state() -> Array:
	var state: Array = []
	for raycast in ObstaclesAndWalls.get_children():
		var collision_point: Vector2 = raycast.get_collision_point()
		state.append(collision_point)
	for raycast in Goals.get_children():
		var collision_point: Vector2 = raycast.get_collision_point()
		state.append(collision_point)
	var obstacle_position: Vector2 = parent_node.obstacle_tile * 64.0
	var goal_position: Vector2 = parent_node.goal_tile * 64.0
	state.append(global_position.distance_to(obstacle_position))
	state.append(global_position.distance_to(goal_position))
	return state
