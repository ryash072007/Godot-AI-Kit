extends Node2D

var cnn: CNN = CNN.new()

func _ready() -> void:
	var next_layer_output: Vector2i = cnn.add_layer(cnn.Layer.SingleFilterConvolutional1D.new(Vector2i(14,14)))
	next_layer_output = cnn.add_layer(cnn.Layer.Pooling.new(next_layer_output))
	next_layer_output = cnn.add_layer(cnn.Layer.Flatten.new(next_layer_output))
	next_layer_output = cnn.add_layer(cnn.Layer.Dense.new(next_layer_output, 9))

	print(cnn.forward(Matrix.rand(Matrix.new(14,14))).data)
