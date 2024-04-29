extends Node2D

var NNA := NeuralNetworkAdvanced.new()

func _ready() -> void:
    NNA.add_layer(2)
    NNA.add_layer(4, NeuralNetworkAdvanced.ACTIVATIONS["sigmoid"])
