extends Node2D

var cnn: CNN = CNN.new()

func _ready() -> void:
	cnn.add_labels(["X", "O"])

	var next_layer_output: Vector2i = cnn.add_layer(cnn.Layer.SingleFilterConvolutional1D.new(Vector2i(14,14)))
	next_layer_output = cnn.add_layer(cnn.Layer.Pooling.new(next_layer_output))
	next_layer_output = cnn.add_layer(cnn.Layer.Flatten.new(next_layer_output))
	next_layer_output = cnn.add_layer(cnn.Layer.Dense.new(next_layer_output, 64))
	next_layer_output = cnn.add_layer(cnn.Layer.SoftmaxDense.new(next_layer_output, 2))

	var pred = cnn.categorise(Matrix.rand(Matrix.new(14,14)))

func _process(_delta: float) -> void:
	var loss = cnn.train(Matrix.rand(Matrix.new(14,14)), "O")
	print(loss)

func load_image_data(image_path: String) -> Matrix:
	var image = Image.new()
	if image.load(image_path) != OK:
		push_error("Failed to load image: " + image_path)
		return Matrix.new(0, 0)
	image.convert(Image.FORMAT_L8)  # Convert to grayscale
	var width = image.get_width()
	var height = image.get_height()
	var matrix = Matrix.new(width, height)
	for x in range(width):
		for y in range(height):
			matrix.data[x][y] = image.get_pixel(x, y).r  # Get the red channel (grayscale)
	return matrix

