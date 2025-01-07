class_name ImageHelper

static func load_grayscale_image_data(image_path: String) -> Matrix:
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

static func load_grayscale_images_from_folder(folder_path: String) -> Array[Matrix]:
	var images: Array[Matrix] = []
	var dir = DirAccess.open(folder_path)
	dir.list_dir_begin()
	var file_name = dir.get_next()
	while file_name != "":
		if !dir.current_is_dir() and file_name.ends_with(".png"):
			var image_path = folder_path.path_join(file_name)
			var image_data = load_grayscale_image_data(image_path)
			images.append(image_data)
		file_name = dir.get_next()
	dir.list_dir_end()
	return images
