extends CharacterBody2D


const SPEED = 350.0
const JUMP_VELOCITY = -350.0

var time_alive: float


var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

var nn: NeuralNetwork


func _ready():
	Engine.max_fps = 24
	for child in $rays.get_children():
		nn.raycasts.append(child)
	$Sprite2d.modulate = nn.color

func _physics_process(delta):
	time_alive += delta
	var addition_arg: Array
	if is_on_floor(): addition_arg = [0.0]
	else: addition_arg = [1.0]
	var prediction = nn.get_prediction_from_raycasts(addition_arg)
	if prediction[0] > 0.5 and is_on_floor():
		velocity.y = JUMP_VELOCITY
	if not is_on_floor():
		velocity.y += gravity * delta

	var direction = 1
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)

	move_and_slide()

	nn.fitness = pow(position.x, 2) / time_alive


func _on_enemy_detector_area_entered(area: Node):
	if area.is_in_group("enemy"):
		queue_free()
