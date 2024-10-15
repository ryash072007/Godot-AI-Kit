extends ColorRect

var heuristic: String = "human"
var speed: float = 200.0

@onready var initial_position: Vector2 = position

@onready var ball: RigidBody2D = $"balancing ball"
@onready var shaft: RigidBody2D = $shaft


func _physics_process(delta: float) -> void:
	if heuristic == "human":
		if Input.is_action_pressed("ui_left"):
			position.x -= speed * delta
		if Input.is_action_pressed("ui_right"):
			position.x += speed * delta


func _on_floor_collision(_body: Node2D) -> void:
	reset()


func reset():
	position = initial_position
	ball.should_reset_position = true
	shaft.should_reset_position = true
