extends Node2D

var NNA := NeuralNetworkAdvanced.new()

func _ready() -> void:
	NNA.add_layer(2)
	NNA.add_layer(4, NNA.ACTIVATIONS["SIGMOID"])

func _physics_process(_delta: float) -> void:
	print(NNA.predict([0, 1]))
