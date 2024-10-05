@tool
extends EditorPlugin

var Neural_Net = load("res://addons/ai_algorithms/Scripts/Neural/Neural_Net.gd")
var logo: CompressedTexture2D = load("res://addons/ai_algorithms/Godot AI Kit.jpg")

func _enter_tree():
	add_custom_type("Neural_Net", "Node2D", Neural_Net, logo)
func _exit_tree():
	remove_custom_type("Neural_Net")
