extends Node2D

var NNA := NeuralNetworkAdvanced.new()

func _ready() -> void:
	NNA.add_layer(2)
	NNA.add_layer(1, NNA.ACTIVATIONS["SIGMOID"])

func _physics_process(_delta: float) -> void:
	# print(NNA.predict([0, 1]))
	NNA.train(
		[[0, 0], [0, 1], [1, 0], [1, 1]],
		[[0], [1], [1], [0]]
		)
