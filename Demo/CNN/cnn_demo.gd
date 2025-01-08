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

var training_O_index: int = 0
var training_X_index: int = 0

var lr_change_rate: float = 0.002


func _ready() -> void:
	cnn.learning_rate = 0.001
	cnn.add_labels(["O", "X"])

	cnn.add_layer(cnn.Layer.MutliFilterConvolutional1D.new(6, "same"))
	cnn.add_layer(cnn.Layer.MultiPoolPooling.new())
	cnn.add_layer(cnn.Layer.Dropout.new(0.1))

	cnn.add_layer(cnn.Layer.MutliFilterConvolutional1D.new(4, "same"))
	cnn.add_layer(cnn.Layer.MultiPoolPooling.new())
	cnn.add_layer(cnn.Layer.Dropout.new(0.1))

	cnn.add_layer(cnn.Layer.Flatten.new())
	cnn.add_layer(cnn.Layer.Dense.new(64, "RELU"))
	cnn.add_layer(cnn.Layer.Dropout.new(0.1))

	cnn.add_layer(cnn.Layer.SoftmaxDense.new(2))

	cnn.compile_network(Vector2i(28, 28), cnn.optimizers.AMSGRAD)

	training_O_images = ImageHelper.load_grayscale_images_from_folder(training_O_dir)
	training_X_images = ImageHelper.load_grayscale_images_from_folder(training_X_dir)
	testing_O_images = ImageHelper.load_grayscale_images_from_folder(testing_O_dir)
	testing_X_images = ImageHelper.load_grayscale_images_from_folder(testing_X_dir)

	training_O_images.shuffle()
	training_X_images.shuffle()


func _physics_process(_delta: float) -> void:
	total_O_loss += cnn.train(training_O_images[training_O_index], "O")
	training_O_index += 1
	if training_O_index >= training_O_images.size():
		training_O_index = 0
		training_O_images.shuffle()

	total_X_loss += cnn.train(training_X_images[training_X_index], "X")
	training_X_index += 1
	if training_X_index >= training_X_images.size():
		training_X_index = 0
		training_X_images.shuffle()

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
			var accuracy: Array[float] = test_all_images()
			var model_accuracy: float = (accuracy[0] + accuracy[1]) / 2
			print("***********************************************")
			print("Model Accuracy: ", model_accuracy)
			print("Model got ", accuracy[0] * testing_O_images.size(), " O out of ", testing_O_images.size(), " correct!")
			print("Model got ", accuracy[1] * testing_X_images.size(), " X out of ", testing_X_images.size(), " correct!")
			print("***********************************************")

			#cnn.learning_rate = max(cnn.learning_rate * (1 - lr_change_rate), 0.0005)
			#print("New Learning Rate: ", cnn.learning_rate)

			if model_accuracy > 0.95:
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

func test_all_images() -> Array[float]:
	var correct_O_predictions: int = 0
	var total_O_predictions: int = 0
	var correct_X_predictions: int = 0
	var total_X_predictions: int = 0


	for image in testing_O_images:
		var prediction = cnn.categorise(image)
		if prediction == "O":
			correct_O_predictions += 1
		total_O_predictions += 1

	for image in testing_X_images:
		var prediction = cnn.categorise(image)
		if prediction == "X":
			correct_X_predictions += 1
		total_X_predictions += 1

	return [float(correct_O_predictions) / float(total_O_predictions), float(correct_X_predictions) / float(total_X_predictions)]

