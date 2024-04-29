extends Node2D

var NNA := NeuralNetworkAdvanced.new()
var i := 0

func _ready() -> void:
	NNA.add_layer(2)
	NNA.add_layer(2, NNA.ACTIVATIONS["SIGMOID"])

func _physics_process(_delta: float) -> void:
	# print(NNA.predict([0, 1]))
	NNA.train(
		[[0, 0], [0, 1], [1, 0], [1, 1]],
		[[1, 0], [0, 1], [0, 1], [1, 0]]
		)
	
	i += 1

	if i >= 200:
		print(NNA.predict([0,0]))
		print(NNA.predict([0,1]))
		print(NNA.predict([1,0]))
		print(NNA.predict([1,1]))
		i = 0
	
