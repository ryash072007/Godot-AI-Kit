# Neural Network and Neural Net Plugin for Godot 4


This plugin allows you to create a Multi Layer Neural Network and also provides a NeuralNet by which you can easily automatically train the network (which can be found under Node2D Section in the add node window).
This plugin is intended for creating AIs that can complete a game level.


## Rules to be followed if using Neural Net

1. If using Neural Net, the identifier or name of the variable of the Neural Network used in your code has to be "nn.". Like this:
```
var nn: NeuralNetwork
```
This is because the Neural Net only works when the Neural Network is named as "nn".

2. If using Neural Net, make sure you do not assign your Neural Network Variable (nn) anything. All you are supposed to do is declare it like this:
```
var nn: NeuralNetwork
```
This is because the Neural Net depends on the fact that "nn" is not assigned anything.

3. When your AI or player has to be killed or removed, always use the "queue_free()" method. This is because the Neural Net relies on the signal emitted by the node when exiting the tree to recieve the fitness and Neural Network of that node. Example:
```
Object.queue_free()
```

## What each variable means and how to use them
1. Ai Scene: This is where you will assign the AI or Player scene by clicking on the drop down arrow on the right side, clicking "quick load" and selecting your scene.
2. Batch Size: This is the informal Batch Size of each generation. The actual batch size of each generation is emitted by the "true_batch_size" signal. This controls the base amount of AIs spawned.
3. Generation Delay: This is the time limit (in seconds) for any generation. Once a generation has lived longer than the amount specified in this, the generation is reset and the next generation comes.
4. Input Nodes: This is where the input nodes for the "nn" will be set. Input Nodes means how many different inputs will the "nn" recieve.
5. Hidden Nodes: This is where the hidden nodes for the "nn" will be set. Hidden Nodes means how many nodes will process the data given by the input nodes. You should experiment with this amount.
6. Output Nodes: This is where you will set how many outputs you want to recieve by the "nn".
7. Random Population: This determines how many AIs with random "nn" will be spawned after the first generation (after the 0 generation). It is a good idea to set this to a value greater than 10 as it allows for more possibilites to be explored by the Neural Net.
8. Use Reproduction: This determines whether reproduction will also be used to create new AIs for the next generations. This enables for combination of different traits of different "nn"s. However, you will most probably not need this as Random and Mutated Population will suffice.
9. Reproduced Population: If "Use Reproduction" is checked, this will determine how many AIs will be spawned with reproduced "nn"s. Note: This value must always be greater than half of the value of Batch Size if you have checked "Use Reproduction" as true.

## How to use Neural Net

Just ensure that all the variables/properties mentioned above are correctly set. The position of this node is where all the AIs will be spawned, meaning, the position of this node = position of AI when spawned.

## How to use Neural Network

```
var nn: NeuralNetwork = NeuralNetwork.new(input_nodes, hidden_nodes, output_nodes)
```
1. Input Nodes: This is where the input nodes for the "nn" will be set. Input Nodes means how many different inputs will the "nn" recieve.
5. Hidden Nodes: This is where the hidden nodes for the "nn" will be set. Hidden Nodes means how many nodes will process the data given by the input nodes. You should experiment with this amount.
6. Output Nodes: This is where you will set how many outputs you want to recieve by the "nn".

7. If the Neural Network depends mostly on inputs from raycasts, you can use the "get_prediction_from_raycasts(optional_val: Array = [])". This function returns an array of floats which are the outputs. The "optional_val" is optional can be used to give more custom inputs in addition to the raycasts. Example:
```
var output = nn.get_prediction_from_raycasts()
# or
var output = nn.get_prediction_from_raycasts([0, 0.4, 2])
```
8. You can use the "predict(input_array: Array[float])" function also to get predictions. Example:
```
var output = nn.predict([0.0, 6, 0.2])
```
9. If you know the expected output of an input, you can use the "train(input_array: Array, target_array: Array)" function in a loop. Example:
```
for epoch in range(2000):
    nn.train([0, 1], [1])
    nn.train([1, 1], [1])
    nn.train([0, 0], [0])
    nn.train([1, 1], [0])
```
10. If you want to mutate your Neural Network, you can do so by:
```
nn = NeuralNetwork.mutate(nn)
```
where nn is your Neural Network.
11. If you want to mutate your Neural Network, you can do so by:
```
new_nn = NeuralNetwork.copy(nn)
```
where nn is your Neural Network and new_nn is the new one to which you are copying the nn to.
12. IF you want to reproduce your Neural Network with another, you can do so by:
```
reproduced_nn = NeuralNetwork.reproduce(nn_1, nn_2)
```
where nn_1 and nn_2 are the parent Neural Networks.

## Credits
I would like to credit Greaby for the basis of the Neural Network.
