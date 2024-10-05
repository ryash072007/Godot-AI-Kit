extends Node2D

var nnas: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()
func _ready() -> void:
	nnas.add_layer(2)
	nnas.add_layer(4, nnas.ACTIVATIONS.ARCTAN)
	nnas.add_layer(6, nnas.ACTIVATIONS.ARCTAN)
	nnas.add_layer(1, nnas.ACTIVATIONS.SIGMOID)
	

func _physics_process(delta: float) -> void:
	nnas.train([0,0], [0])
	nnas.train([1,0], [1])
	nnas.train([0,1], [1])
	nnas.train([1,1], [0])
	
	if Input.is_action_just_pressed("predict"):
		print("--------------Prediction--------------")
		print(nnas.predict([0,0]))
		print(nnas.predict([1,0]))
		print(nnas.predict([0,1]))
		print(nnas.predict([1,1]))
