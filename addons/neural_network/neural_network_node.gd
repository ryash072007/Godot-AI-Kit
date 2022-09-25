extends NeuralNetwork

func _enter_tree():
	assert(learning_rate != 0, "NN's Learning Rate can NOT be 0!")
	assert(input_nodes != 0 or output_nodes != 0, "Please set the NN's input or output nodes from the Inspector tab!")
	nodes_set = true
