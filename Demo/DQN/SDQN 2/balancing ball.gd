extends RigidBody2D

@onready var ball_initial_position: Vector2 = position
var should_reset_position: bool = false

func _integrate_forces(state):
	if should_reset_position:
		state.transform.origin = ball_initial_position
		should_reset_position = false
