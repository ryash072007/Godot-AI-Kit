@tool
extends EditorPlugin

var Neural_Net = load("res://addons/ai_algorithms/Scripts/Neural/Neural_Net.gd")
var SDQN_Node = load("res://addons/ai_algorithms/Scripts/DQN/SDQN_Node.gd")
var logo: CompressedTexture2D = load("res://addons/ai_algorithms/Godot AI Kit.jpg")

func _enter_tree():
	add_custom_type("Neural Net", "Node2D", Neural_Net, logo)
	add_custom_type("Simple DQN", "Node2D", SDQN_Node, logo)

func _exit_tree():
	remove_custom_type("Neural_Net")
	remove_custom_type("SDQN_Node")
