@tool
extends EditorPlugin


func _enter_tree():
	add_custom_type("NeuralNetworkNode", "Node", load("res://addons/neural_network/neural_network_node.gd"), load("res://addons/neural_network/NeuralNetwork.png"))


func _exit_tree():
	remove_custom_type("NeuralNetworkNode")
