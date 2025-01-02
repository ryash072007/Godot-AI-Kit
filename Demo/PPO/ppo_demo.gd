extends Node2D

@onready var WorldTileMap: TileMap = $WorldTileMap
var top_left_corner: Vector2i = Vector2i(5,1)
var bottom_right_corner: Vector2i = Vector2i(12,8)
enum tile {OBSTACLE, GOAL, WALL}

var obstacle_tile: Vector2i
var goal_tile: Vector2i

func _ready() -> void:
	reset_env()

func reset_env() -> void:
	obstacle_tile = get_valid_random_position()

	while true:
		goal_tile = get_valid_random_position()
		if obstacle_tile != goal_tile:
			break


	WorldTileMap.set_cell(0, obstacle_tile, tile.OBSTACLE, Vector2i.ZERO, 1)
	WorldTileMap.set_cell(0, goal_tile, tile.GOAL, Vector2i.ZERO, 1)

func get_valid_random_position() -> Vector2i:
	var pos: Vector2i
	while true:
		pos = Vector2i(
				randi_range(top_left_corner.x, bottom_right_corner.x),
				randi_range(top_left_corner.y, bottom_right_corner.y)
				)
		if not (pos.x in [8,9] and pos.y in [4,5]):
			break
	return pos
