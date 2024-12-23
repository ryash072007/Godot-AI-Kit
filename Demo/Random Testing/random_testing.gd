extends Node2D

var a: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new(NeuralNetworkAdvanced.methods.ADAM)

func _ready() -> void:
	var blah := test.new()
	var asdas := Callable(test, "function")
	#a.add_layer(4)
	#a.add_layer(2, "LINEAR")
	#print(a.predict([10,10,10,10]))
	#a.save("user://test.blah")
	#var v := NeuralNetworkAdvanced.new()
	#v.load("user://test.blah")
	#EncodedObjectAsID
	#print(v.predict([10,10,10,10]))
	var time := Time.get_ticks_usec()
	asdas.call()
	time = Time.get_ticks_usec() - time
	print(time)
	time = Time.get_ticks_usec()
	blah.get("function").call()
	time = Time.get_ticks_usec() - time
	print(time)

	time = Time.get_ticks_usec()
	asdas.call()
	time = Time.get_ticks_usec() - time
	print(time)
	time = Time.get_ticks_usec()
	blah.get("function").call()
	time = Time.get_ticks_usec() - time
	print(time)

class test:
	static func function():
		print("hmm")
