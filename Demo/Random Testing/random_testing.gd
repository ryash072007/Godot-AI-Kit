extends Node2D


func _ready() -> void:
	var data: Matrix = Matrix.new(4,4)
	data.data[0][0] = 1
	data.data[3][3] = 2

	data = Augmentations.rotate_random(data)
	print(data.data)

class test:
	static func function():
		print("hmm")
