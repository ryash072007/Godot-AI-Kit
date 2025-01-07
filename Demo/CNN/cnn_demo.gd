extends Node2D

var cnn: CNN = CNN.new()

var training_O_dir: String = "res://Demo/CNN/training_data/O/"
var training_X_dir: String = "res://Demo/CNN/training_data/X/"

var testing_O_dir: String = "res://Demo/CNN/testing_data/O/"
var testing_X_dir: String = "res://Demo/CNN/testing_data/X/"

var training_O_images: Array[Matrix] = []
var training_X_images: Array[Matrix] = []
var testing_O_images: Array[Matrix] = []
var testing_X_images: Array[Matrix] = []

var training_steps: int = 0
var total_O_loss: float = 0.0
var total_X_loss: float = 0.0

func _ready() -> void:
	cnn.learning_rate = 0.005
	cnn.add_labels(["O", "X"])

	var next_layer_output = cnn.add_layer(cnn.Layer.SingleFilterConvolutional1D.new(Vector2i(14,14)))
	next_layer_output = cnn.add_layer(cnn.Layer.Pooling.new(next_layer_output))
	next_layer_output = cnn.add_layer(cnn.Layer.Flatten.new(next_layer_output))  # 8 feature maps
	next_layer_output = cnn.add_layer(cnn.Layer.Dense.new(next_layer_output, 16))
	next_layer_output = cnn.add_layer(cnn.Layer.SoftmaxDense.new(next_layer_output, 2))

	training_O_images = load_images_from_folder(training_O_dir)
	training_X_images = load_images_from_folder(training_X_dir)
	testing_O_images = load_images_from_folder(testing_O_dir)
	testing_X_images = load_images_from_folder(testing_X_dir)

func _physics_process(_delta: float) -> void:
	if training_O_images.size() > 0:
		total_O_loss += cnn.train(training_O_images.pick_random(), "O")

	if training_X_images.size() > 0:
		total_X_loss += cnn.train(training_X_images.pick_random(), "X")

	training_steps += 1

	if training_steps % 100 == 0:
		var avg_O_loss = total_O_loss / 100.0
		var avg_X_loss = total_X_loss / 100.0
		total_O_loss = 0.0
		total_X_loss = 0.0

		print("________________________________________________________")
		print("Average O Loss: ", avg_O_loss, "at training step: ", training_steps)
		print("Average X Loss: ", avg_X_loss, "at training step: ", training_steps)

		var accuracy = test_all_images()
		print("Model Accuracy: ", accuracy, "%")

		if accuracy > 98.0:
			print("Training complete. Accuracy is greater than 98%.")
			get_tree().quit()

	if Input.is_action_just_pressed("ui_accept"):
		print("________________________________________________________")
		if testing_O_images.size() > 0:
			var test_image_O = testing_O_images.pick_random()
			var prediction_O = cnn.categorise(test_image_O)
			print("Testing O - Prediction: ", prediction_O)

		if testing_X_images.size() > 0:
			var test_image_X = testing_X_images.pick_random()
			var prediction_X = cnn.categorise(test_image_X)
			print("Testing X - Prediction: ", prediction_X)
		print("________________________________________________________")

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

func load_images_from_folder(folder_path: String) -> Array[Matrix]:
	var images: Array[Matrix] = []
	var dir = DirAccess.open(folder_path)
	dir.list_dir_begin()
	var file_name = dir.get_next()
	while file_name != "":
		if !dir.current_is_dir() and file_name.ends_with(".png"):
			var image_path = folder_path.path_join(file_name)
			var image_data = load_image_data(image_path)
			images.append(image_data)
		file_name = dir.get_next()
	dir.list_dir_end()
	return images

func test_all_images() -> float:
	var correct_predictions: int = 0
	var total_predictions: int = 0

	for image in testing_O_images:
		var prediction = cnn.categorise(image)
		if prediction == "O":
			correct_predictions += 1
		total_predictions += 1

	for image in testing_X_images:
		var prediction = cnn.categorise(image)
		if prediction == "X":
			correct_predictions += 1
		total_predictions += 1

	var accuracy = float(correct_predictions) / float(total_predictions) * 100.0
	return accuracy

