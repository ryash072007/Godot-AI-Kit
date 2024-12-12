extends RigidBody2D

var should_reset_position: bool = false
@onready var shaft_initial_position: Vector2 = position

func reset() -> void:
		rotation_degrees = 0
		linear_velocity = Vector2.ZERO
		angular_velocity = 0
		transform.origin = shaft_initial_position
		should_reset_position = false
