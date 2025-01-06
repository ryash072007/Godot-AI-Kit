class_name CNN

class Layer:
    class SingleFilterConvolutional1D:
        var filter: Matrix
        var bias: float = 0.0
        var stride: int = 1
        var input_shape: Vector2i = Vector2(0, 0)
        var filter_shape: Vector2i = Vector2(0, 0)
        var output_shape: Vector2i = Vector2(0, 0)
        var output: Matrix

        var activationFunction := Activation.RELU

        func _init(_input_shape: Vector2i, _output_shape: Vector2i, _filter_shape: Vector2i = Vector2i(3,3), _stride: int = 1) -> void:
            input_shape = _input_shape
            output_shape = _output_shape
            filter_shape = _filter_shape
            stride = _stride
            filter = Matrix.uniform_he_init(Matrix.new(filter_shape.x, filter_shape.y), filter_shape.x * filter_shape.y)
    
    class Pooling:
        var stride: int = 1
        var pool_size: Vector2i = Vector2i(0, 0)
        var input_shape: Vector2i = Vector2i(0, 0)
        var output_shape: Vector2i = Vector2i(0, 0)
        var output: Matrix

        func _init(_input_shape: Vector2i, _output_shape: Vector2i, _pool_size: Vector2i = Vector2i(2, 2), _stride: int = 2) -> void:
            input_shape = _input_shape
            output_shape = _output_shape
            pool_size = _pool_size
            stride = _stride
    
    class Dense:
        var weights: Matrix
        var biases: Matrix
        var activation: String = "RELU"
        var input_shape: int = 0
        var output_shape: int = 0
        var output: Matrix

        var ACTIVATIONS: Activation = Activation.new()
        var activationFunction

        func _init(_input_shape: int, _output_shape: int, _activation: String = "RELU") -> void:
            input_shape = _input_shape
            output_shape = _output_shape
            activation = _activation
        
            activationFunction = ACTIVATIONS.get(activation)

            if activation in ["RELU", "LEAKYRELU", "ELU", "LINEAR"]:
                weights = Matrix.uniform_he_init(Matrix.new(output_shape, input_shape), input_shape)
            elif activation in ["SIGMOID", "TANH"]:
                weights = Matrix.uniform_glorot_init(Matrix.new(output_shape, input_shape), input_shape, output_shape)
            else:
                weights = Matrix.rand(Matrix.new(output_shape, input_shape))
            
            biases = Matrix.new(output_shape, 1)
        
        # Different from NNA as there input was transposed and so dot product was used
        func forward(input: Matrix) -> Matrix:
            output = Matrix.add(Matrix.multiply(input, weights), biases)
            output = Matrix.map(output, activationFunction.function)
            return output
        
        

