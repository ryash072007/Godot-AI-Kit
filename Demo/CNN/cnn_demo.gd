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

var total_testing_images: int

func _ready() -> void:
	cnn.learning_rate = 0.001
	cnn.add_labels(["O", "X"])

	cnn.add_layer(cnn.Layer.MutliFilterConvolutional1D.new(4, "same"))
	cnn.add_layer(cnn.Layer.BatchNormalization.new())
	cnn.add_layer(cnn.Layer.MultiPoolPooling.new())
	#cnn.add_layer(cnn.Layer.MutliFilterConvolutional1D.new(4, "same"))
	#cnn.add_layer(cnn.Layer.MultiPoolPooling.new())
	cnn.add_layer(cnn.Layer.Flatten.new())
	cnn.add_layer(cnn.Layer.Dense.new(64, "RELU"))
	cnn.add_layer(cnn.Layer.SoftmaxDense.new(2))

	cnn.compile_network(Vector2i(28, 28))

	training_O_images = ImageHelper.load_grayscale_images_from_folder(training_O_dir)
	training_X_images = ImageHelper.load_grayscale_images_from_folder(training_X_dir)
	testing_O_images = ImageHelper.load_grayscale_images_from_folder(testing_O_dir)
	testing_X_images = ImageHelper.load_grayscale_images_from_folder(testing_X_dir)

func _process(_delta: float) -> void:
	if training_O_images.size() > 0:
		total_O_loss += cnn.train(training_O_images.pick_random(), "O")

	if training_X_images.size() > 0:
		total_X_loss += cnn.train(training_X_images.pick_random(), "X")

	training_steps += 1

	if training_steps % 50 == 0:
		var avg_O_loss = total_O_loss / 100.0
		var avg_X_loss = total_X_loss / 100.0
		total_O_loss = 0.0
		total_X_loss = 0.0

		print("________________________________________________________")
		print("Average O Loss: ", avg_O_loss, " at training step: ", training_steps)
		print("Average X Loss: ", avg_X_loss, " at training step: ", training_steps)

		if training_steps % 100 == 0:
			var accuracy = test_all_images()
			print("Model Accuracy: ", accuracy)
			print("Model got ", accuracy * total_testing_images, " out of ", total_testing_images, " correct!")

			if accuracy > 0.95:
				print("Training complete. Accuracy is greater than 95%.")
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

func test_all_images() -> float:
	var correct_predictions: int = 0
	var total_predictions: int = 0

	total_testing_images = testing_O_images.size() + testing_X_images.size()

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

	var accuracy = float(correct_predictions) / float(total_predictions)
	return accuracy

