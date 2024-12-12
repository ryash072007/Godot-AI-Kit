extends ColorRect

var heuristic: String = "human"
var speed: float = 200.0

@onready var initial_position: Vector2 = position
@onready var shaft: RigidBody2D = $shaft
var was_just_reset: bool = false

func _physics_process(delta: float) -> void:
	if was_just_reset:
		was_just_reset = false
		$shaft.set_deferred("freeze", false)
	if heuristic == "human":
		if Input.is_action_pressed("ui_left"):
			position.x -= speed * delta
		if Input.is_action_pressed("ui_right"):
			position.x += speed * delta


func _on_floor_collision(_body: Node2D) -> void:
	reset()


func reset():
	$shaft.set_deferred("freeze", true)
	shaft.reset()
	position = initial_position
	was_just_reset = true
