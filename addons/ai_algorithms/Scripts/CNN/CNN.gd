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

        var input: Matrix

        var activationFunction := Activation.RELU

        func _init(_input_shape: Vector2i, _output_shape: Vector2i, _filter_shape: Vector2i = Vector2i(3,3), _stride: int = 1) -> void:
            input_shape = _input_shape
            output_shape = _output_shape
            filter_shape = _filter_shape
            stride = _stride
            filter = Matrix.uniform_he_init(Matrix.new(filter_shape.x, filter_shape.y), filter_shape.x * filter_shape.y)
        
        func forward(_input: Matrix) -> Matrix:
            input = _input
            output = Matrix.new(output_shape.x, output_shape.y)
            for i in range(0, input_shape.x - filter_shape.x + 1, stride):
                for j in range(0, input_shape.y - filter_shape.y + 1, stride):
                    var sum = 0.0
                    for x in range(filter_shape.x):
                        for y in range(filter_shape.y):
                            sum += input.data[i + x][j + y] * filter.data[x][y]
                    output.data[i / stride][j / stride] = sum + bias
            output = Matrix.map(output, activationFunction.function)
            return output
        
        func backward(dout: Matrix) -> Dictionary:
            var dW: Matrix = Matrix.new(filter_shape.x, filter_shape.y)
            var dB: float = 0.0
            var dX: Matrix = Matrix.new(input_shape.x, input_shape.y)
            for i in range(0, input_shape.x - filter_shape.x + 1, stride):
                for j in range(0, input_shape.y - filter_shape.y + 1, stride):
                    for x in range(filter_shape.x):
                        for y in range(filter_shape.y):
                            dW.data[x][y] += input.data[i + x][j + y] * dout.data[i / stride][j / stride]
                            dX.data[i + x][j + y] += filter.data[x][y] * dout.data[i / stride][j / stride]
                    dB += dout.data[i / stride][j / stride]
            return {
                "dW": dW,
                "dB": dB,
                "dX": dX
            }
    
    class Pooling:
        var stride: int = 1
        var pool_size: Vector2i = Vector2i(0, 0)
        var input_shape: Vector2i = Vector2i(0, 0)
        var output_shape: Vector2i = Vector2i(0, 0)
        var output: Matrix

        # Storing the input from the last forward pass
        var input: Matrix

        func _init(_input_shape: Vector2i, _pool_size: Vector2i = Vector2i(2, 2), _stride: int = 2) -> void:
            input_shape = _input_shape
            pool_size = _pool_size
            stride = _stride

            output_shape = Vector2i(
                (input_shape.x - pool_size.x) / stride + 1,
                (input_shape.y - pool_size.y) / stride + 1
            )
        
        # Max pooling
        func forward(_input: Matrix) -> Matrix:
            input = _input
            output = Matrix.new(output_shape.x, output_shape.y)
            for i in range(0, input_shape.x - pool_size.x + 1, stride):
                for j in range(0, input_shape.y - pool_size.y + 1, stride):
                    var max_val = -INF
                    for x in range(pool_size.x):
                        for y in range(pool_size.y):
                            max_val = max(max_val, input.data[i + x][j + y])
                    output.data[i / stride][j / stride] = max_val
            return output
        
        func backward(dout: Matrix) -> Matrix:
            var dx = Matrix.new(input_shape.x, input_shape.y)
            for i in range(0, input_shape.x - pool_size.x + 1, stride):
                for j in range(0, input_shape.y - pool_size.y + 1, stride):
                    var max_val = -INF
                    var max_x = 0
                    var max_y = 0
                    for x in range(pool_size.x):
                        for y in range(pool_size.y):
                            if input.data[i + x][j + y] > max_val:
                                max_val = input.data[i + x][j + y]
                                max_x = i + x
                                max_y = j + y
                    dx.data[max_x][max_y] = dout.data[i / stride][j / stride]
            return dx

    
    class Dense:
        var weights: Matrix
        var biases: Matrix
        var activation: String = "RELU"
        var input_shape: int = 0
        var output_shape: int = 0
        var output: Matrix

        # Storing the input from the last forward pass
        var input: Matrix

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
        func forward(_input: Matrix) -> Matrix:
            input = _input
            output = Matrix.add(Matrix.multiply(input, weights), biases)
            output = Matrix.map(output, activationFunction.function)
            return output
        
        func backward(dout: Matrix) -> Dictionary:
            var dW: Matrix = Matrix.outer_product(dout, input)
            var dB: Matrix = dout
            var dX: Matrix = Matrix.multiply(weights, dout)
            return {
                "dW": dW,
                "dB": dB,
                "dX": dX
            }

