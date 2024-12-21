extends Node2D

@export var agents_to_spawn: int = 3
@onready var agent = preload("res://Demo/DQN/SDQN/agent.tscn")

func _ready() -> void:
	for i in range(agents_to_spawn):
		var agent = agent.instantiate()
		if i % 2 == 0:
			agent.check_fitness = false

		call_deferred("add_child", agent)
