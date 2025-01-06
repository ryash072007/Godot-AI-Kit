class_name CNN

class Layer:
    class Convolutional:
        var filters = []
        var biases = []
        var stride = 1
        var padding = 0
        var activation = "relu"
        var input_shape = Vector2(0, 0)
        var filter_shape = Vector2(0, 0)
        var output_shape = Vector2(0, 0)
        var output = []
    
    class Pooling:
        var stride = 1
        var padding = 0
        var pool_size = Vector2(0, 0)
        var input_shape = Vector2(0, 0)
        var output_shape = Vector2(0, 0)
        var output = []
    
    class Dense:
        var weights = []
        var biases = []
        var activation = "relu"
        var input_shape = 0
        var output_shape = 0
        var output = []
    
    