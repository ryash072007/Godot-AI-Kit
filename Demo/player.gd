extends CharacterBody2D


const SPEED = 150.0
const JUMP_VELOCITY = -350.0

var nn_set := false
# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

var top_level_node

var NN = NeuralNetwork.new()
var fitness: float

func _ready():
	if !nn_set: NN.set_nn_data(3, 5, 1)
	for child in $rays.get_children():
		NN.raycasts.append(child)
	$Sprite2d.modulate = NN.color
	top_level_node = get_tree().get_first_node_in_group("demo")

func _physics_process(delta):
	var addition_arg: Array
	if is_on_floor(): addition_arg = [0.0]
	else: addition_arg = [1.0]
	
	
	var prediction = NN.get_prediction_from_raycasts(addition_arg)
#	print(prediction)
	if prediction[0] > 0.6 and is_on_floor():
		velocity.y = JUMP_VELOCITY
	# Add the gravity.
	if not is_on_floor():
		velocity.y += gravity * delta

	# Handle Jump.
#	if Input.is_action_pressed("ui_accept") and is_on_floor():
#		velocity.y = JUMP_VELOCITY

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	var direction = 1
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)
	
	move_and_slide()


func _on_dec_area_entered(area):
	if area.name == "kill":
		top_level_node.emit_signal("on_death", position.x, NN)
		call_deferred("queue_free")
