@tool
extends EditorPlugin

var Neural_Net = load("res://addons/neural_net/Scripts/Neural_Net.gd")
var logo: CompressedTexture2D = load("res://addons/neural_net/NeuralNetwork.png")

func _enter_tree():
	add_custom_type("Neural_Net", "Node2D", Neural_Net, logo)
func _exit_tree():
	remove_custom_type("Neural_Net")
