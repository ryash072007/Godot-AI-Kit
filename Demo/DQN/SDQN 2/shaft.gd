extends RigidBody2D

var should_reset_position: bool = false
@onready var shaft_initial_position: Vector2 = position

func _integrate_forces(state):
	if should_reset_position:
		state.transform.origin = shaft_initial_position
		rotation_degrees = 0
		should_reset_position = false
