extends Node2D

signal on_death

var ai = preload("res://Demo/player.tscn")


var best_fitness: float
var best_nn

var batch_size := 25
var left_alive: int

var generation: int = 0

func _ready():
#	Engine.time_scale = 2
	left_alive = batch_size
	on_death.connect(say_death)
	spawn()



func say_death(fitness, NN):
	left_alive -= 1
	if fitness > best_fitness:
		best_fitness = fitness
		best_nn = NeuralNetwork.copy(NN)
#	print(fitness)


func _process(_delta):
	$gen.text = str(generation - 1)
	$al.text = str(left_alive)
	$fit.text = str(best_fitness)
	if $AIs.get_child_count() == 0:
#		print("Gen " + str(generation - 1) + ", Best Fitness: " + str(best_fitness))
		left_alive = batch_size
		spawn(batch_size - 1)
		var new = ai.instantiate()
		new.position = $pos.position
		new.NN = best_nn
		new.nn_set = true
		$AIs.add_child(new)
	
func spawn(amt: int = batch_size):
	for i in range(amt):
		var new = ai.instantiate()
		new.position = $pos.position
		if generation != 0: new.NN = NeuralNetwork.mutate(best_nn)
		if generation != 0: new.nn_set = true
		$AIs.add_child(new)
	generation += 1

#
#func _on_won_body_entered(body):
#	if body.is_in_group("ai"):
#		print(body.NN)
