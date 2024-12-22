extends Node2D

var a: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()

func _ready() -> void:
	a.add_layer(2)
	a.add_layer(2)
	var b: NeuralNetworkAdvanced = a.copy()
	a.network[0] = "NAHOM"
	print(a.network)
	print(b.network)
