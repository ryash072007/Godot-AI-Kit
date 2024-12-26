TODO:

~~1. Multi-Layer Neural Network Support (more than a single hidden network)~~ [Completed]

~~2. PPO Support (Unrealistic but will try!)~~ [Might start soon]

~~3. Simple DQN Support~~ [Completed]



# AI Algorithm for Godot 4

The goal of this project is to provide a variety of AI Algorithms in Godot 4 natively using GDscript.

## Index

1. [Simple Neural Network and Neural Net](#simple-neural-network-and-neural-net-plugin-for-godot-4)
2. [Neural Network Advanced (Multi-Layered Neural Network)](#NNA)
3. [Q-Learning Algorithm (and SARSA)](#q-learning-algorithm)
4. [Minimax Algorithm](#minimax-algorithm)

## Simple Neural Network and Neural Net Plugin for Godot 4

This part of the plugin allows you to create a Multi Layer Neural Network and also provides a NeuralNet by which you can easily automatically train the network (which can be found under Node2D Section in the add node window).  
This plugin is intended for creating AIs that can complete a game level.

### Rules to be followed if using Neural Net

1. If using Neural Net, the identifier or name of the variable of the Neural Network used in your code has to be `nn`. Like this:

    ```gdscript
    var nn: NeuralNetwork
    ```

    This is because the Neural Net only works when the Neural Network is named as `nn`.

2. If using Neural Net, make sure you do not assign your Neural Network Variable `nn` anything. All you are supposed to do is declare it like this:

    ```gdscript
    var nn: NeuralNetwork
    ```

    This is because the Neural Net depends on the fact that `nn` is not assigned anything.

3. When your AI or player has to be killed or removed, always use the `queue_free()` method. This is because the Neural Net relies on the signal emitted by the node when exiting the tree to receive the fitness and Neural Network of that node. Example:

    ```gdscript
    Object.queue_free()
    ```

### What each variable means and how to use them

1. **Ai Scene**: This is where you will assign the AI or Player scene by clicking on the drop down arrow on the right side, clicking `quick load` and selecting your scene.
2. **Batch Size**: This is the informal Batch Size of each generation. The actual batch size of each generation is emitted by the `true_batch_size` signal. This controls the base amount of AIs spawned.
3. **Generation Delay**: This is the time limit (in seconds) for any generation. Once a generation has lived longer than the amount specified in this, the generation is reset and the next generation comes.
4. **Input Nodes**: This is where the input nodes for the `nn` will be set. Input Nodes means how many different inputs will the `nn` receive.
5. **Hidden Nodes**: This is where the hidden nodes for the `nn` will be set. Hidden Nodes means how many nodes will process the data given by the input nodes. You should experiment with this amount.
6. **Output Nodes**: This is where you will set how many outputs you want to receive by the `nn`.
7. **Random Population**: This determines how many AIs with random `nn` will be spawned after the first generation (after the 0 generation). It is a good idea to set this to a value greater than 10 as it allows for more possibilities to be explored by the Neural Net.
8. **Use Reproduction**: This determines whether reproduction will also be used to create new AIs for the next generations. This enables for combination of different traits of different `nn`s. However, you will most probably not need this as Random and Mutated Population will suffice.
9. **Reproduced Population**: If “Use Reproduction” is checked, this will determine how many AIs will be spawned with reproduced `nn`s. Note: This value must always be greater than half of the value of Batch Size if you have checked “Use Reproduction” as true.

### How to use Neural Net

Just ensure that all the variables/properties mentioned above are correctly set. The position of this node is where all the AIs will be spawned, meaning, the position of this node = position of AI when spawned.

### How to use Neural Network

```gdscript
var nn: NeuralNetwork = NeuralNetwork.new(input_nodes, hidden_nodes, output_nodes)
```

1. **Input Nodes**: This is where the input nodes for the `nn` will be set. Input Nodes means how many different inputs will the `nn` receive.
2. **Hidden Nodes**: This is where the hidden nodes for the `nn` will be set. Hidden Nodes means how many nodes will process the data given by the input nodes. You should experiment with this amount.
3. **Output Nodes**: This is where you will set how many outputs you want to receive by the `nn`.
4. If the Neural Network depends mostly on inputs from raycasts, you can use the `get_prediction_from_raycasts(optional_val: Array = [])`. This function returns an array of floats which are the outputs. The `optional_val` is optional can be used to give more custom inputs in addition to the raycasts. Example:

    ```gdscript
    var output = nn.get_prediction_from_raycasts()

    # or

    var output = nn.get_prediction_from_raycasts([0, 0.4, 2])
    ```

5. You can use the `predict(input_array: Array[float])` function also to get predictions. Example:

    ```gdscript
    var output = nn.predict([0.0, 6, 0.2])
    ```

6. If you know the expected output of an input, you can use the `train(input_array: Array, target_array: Array)` function in a loop. Example:

    ```gdscript
    for epoch in range(2000):
        nn.train([0, 1], [1])
        nn.train([1, 1], [1])
        nn.train([0, 0], [0])
        nn.train([1, 1], [0])
    ```

7. If you want to mutate your Neural Network, you can do so by:

    ```gdscript
    nn = NeuralNetwork.mutate(nn)
    ```

    where `nn` is your Neural Network.

8. If you want to copy your Neural Network, you can do so by:

    ```gdscript
    new_nn = NeuralNetwork.copy(nn)
    ```

    where `nn` is your Neural Network and `new_nn` is the new one to which you are copying the `nn` to.

9. If you want to reproduce your Neural Network with another, you can do so by:

    ```gdscript
    reproduced_nn = NeuralNetwork.reproduce(nn_1, nn_2)
    ```

    where `nn_1` and `nn_2` are the parent Neural Networks.

## Neural Network Advanced

**Note:** Support for this in the Neural Net has not been implemented yet.

### How to use NeuralNetworkAdvanced class

1. Initialising the NNA variable

    ```gdscript
    var nnas: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()
    ```

2. Add the first layer to the network. Here you should only specify the number of nodes needed in this layer.

    ```gdscript
    nnas.add_layer(2)
    ```

3. Add the remaining layers to the network. Here you can also specify which activation function to use. Eg:

    ```gdscript
    nnas.add_layer(4, Activation.ARCTAN)
    nnas.add_layer(6, Activation.ARCTAN)
    nnas.add_layer(1, Activation.SIGMOID)
    ```

	Currently available activation functions are listed in this file: [Activation.gd](https://github.com/ryash072007/Godot-AI-Kit/blob/main/addons/ai_algorithms/Scripts/Neural/Activation.gd)


4. To train the network, you can call the `train()` function on the NNA. The first argument has to be the **input array of size same as that of the first layer** and the **second argument has to be the output array of size same as the last layer of the network.**  
    Note: This only runs a single train call. You need to do a lot of these to train your NNA to be accurate. Eg: Training for an XOR Gate. In the demo, you can see that this code is placed in the `_physics_process` function so that it is ran many times a second.

    ```gdscript
    nnas.train([0,0], [0])
    nnas.train([1,0], [1])
    nnas.train([0,1], [1])
    nnas.train([1,1], [0])
    ```

5. To get a prediction/output from the NNA. You have to call the `predict` function on the NNA. The first and only argument has to be input array for the network. It will return an array of the same size as that of the last/output layer. Eg:

    ```gdscript
    print(nnas.predict([1,0]))
    ```

    will return `[1]` when trained.



### Configurable parameters in NeuralNetworkAdvanced

1. Learning Rate: Default value is `0.001`
	```gdscript
	nnas.learning_rate = 0.001 # Or any other float
	```

2. Backward Propagation Optimiser Method: Currently available methods are SGD (Scholastic Gradient Descent) [<- DEFAULT] and ADAM (Adaptive Moment Estimation). Has to be given as the argument during NeuralNetworkAdvanced.new() call.
	```gdscript
	var nnas := NeuralNetworkAdvanced.new(NeuralNetworkAdvanced.methods.SGD)
	--or--
	var nnas := NeuralNetworkAdvanced.new(NeuralNetworkAdvanced.methods.ADAM)
	```

3. Clip Value: Only if optimiser method is chosen as SGD. Default value is `INF`.
	```gdscript
	nnas.clip_value = 10.0 # Or any other value you want to normalise/clip the gradients to
	```

4. Use AmsGrad: Only if optimiser method is chosen as ADAM. Set to true if you want more stable training. Default value is `false`.
	```gdscript
	nnas.use_amsgrad = true
	```

### Additional Methods in NeuralNetworkAdvanced

1. **copy(all: bool = false)**: Creates a deep copy of the neural network. If `all` is true, copies all properties; if false, copies only essential properties.
    ```gdscript
    var copied_nna: NeuralNetworkAdvanced = nnas.copy()
    ```

2. **to_dict()**: Serializes the neural network to a dictionary. Used for saving the network state.
    ```gdscript
    var data: Dictionary = nnas.to_dict()
    ```

3. **from_dict(dict: Dictionary)**: Deserializes the neural network from a dictionary. Used for loading the network state.
    ```gdscript
    nnas.from_dict(dict)
    ```

4. **save(file_path: String)**: Saves the neural network state to a file.
    ```gdscript
    nnas.save("res://path/to/save/file.json")
    ```

### Note

1. Addition of layers should only happen once and so `_ready()` is an appropriate place to put them.
2. For more detailed ADAM hyperparameters, check [Neural_Network_Advanced.gd](https://github.com/ryash072007/Godot-AI-Kit/blob/main/addons/ai_algorithms/Scripts/Neural/Neural_Network_Advanced.gd)




## Q-Learning Algorithm

This algorithm implements Q-Learning algorithm using Q-Table natively in Godot.

### How to use QLearning class

1. Initialise a QLearning variable

    ```gdscript
    var qnet: QLearning = QLearning.new(observation_space, action_space, is_learning, not_sarsa)
    ```

    Both the `observation_space` and `action_space` have to be natural numbers representing the possible states the agent can be in and the possible actions choices the agent can take in any given state. `is_learning` is a boolean value of whether the agent should be learning or not, and `not_sarsa` set to `true` will disable sarsa (on-policy). I would recommend sarsa if you want a safer route to the final path.

2. Get a prediction from the QLearning variable:

    ```gdscript
    qnet.predict(current_state, reward_of_previous_state)
    ```

    The above method returns an whole number that lies between `0` and `action_space - 1`. The value returned corresponds to an action the agent can take.  
    You can assign the returned value to variable by:

    ```gdscript
    var action_to_do: int = qnet.predict(current_state, previous_reward)
    ```

### Configurable Values

1. `qnet.exploration_probability` -> has to be a float value  
    **Default Value: `1.0`**  
    The probability that the agent will take a random action or exploit the data it has learned.  
    Do not change unless you know what you are doing.

2. `qnet.exploration_decreasing_decay` -> has to be a float value  
    **Default Value: `0.01`**  
    Changes how the value by which the `qnet.exploration_probability` decreases every `qnet.decay_per_steps` steps.

3. `qnet.min_exploration_probability` -> has to be a float value  
    **Default Value: `0.01`**  
    The minimum value the `exploration_probability` can take.

4. `qnet.learning_rate` -> has to be a float  
    **Default Value: `0.2`**  
    The rate at which the agent learns.

5. `qnet.decay_per_steps` -> has to be natural number  
    **Default Value: `100`**  
    After how many steps does the `qnet.exploration_probability` decrease by `qnet.exploration_decreasing_decay` value.

6. `qnet.is_learning` -> has to be a bool value  
    **Default Value: `true`**  
    To be set to false only when the `qnet.QTable.data` is set manually.

7. `print_debug_info` -> has to be a bool value  
    **Default Value: `false`**  
    This can be set to `true` if you want to print debug info - total steps completed and current exploration probability - every `qnet.decay_per_steps`.

### Things to keep in mind when using QLearning

1. The predict method of the QLearning class takes two compulsory arguments:

    ```gdscript
    qnet.predict(current_state, previous_state_reward)
    ```

    The `current_state` has to be a whole number representing the state it is currently in, while the `previous_state_reward` has to a float representing the reward it got for the previous action it took.

## Minimax Algorithm

Alpha-Beta Pruning has been implemented!  
If the AI is playing the role of the adversary, then set `minimax.is_adversary` to `true` else `false`.

### How to use Minimax class

1. Initialise the Minimax class with 4 arguments:

    ```gdscript
    var minimax: Minimax = Minimax.new(Callable(result), Callable(terminal), Callable(utility), Callable(possible_actions))
    ```

    1. `result_func: Callable`: This callable argument must link to the function in your code that returns the state of the environment after a particular action is performed.
    2. `terminal_func: Callable`: This callable argument must link to the function in your code that returns `true` if the game is over and `false` if the game can continue for a given state.
    3. `utility_func: Callable`: This callable argument must link to the function in your code that returns the value of the given state. Currently this function only runs when the game is a terminal state. Losing states should have lesser value than winning states.
    4. `possible_actions_func: Callable`: This callable argument must link to the function in your code that returns all the possible actions for a given state.

2. Every time the AI needs to perform an action, call the `action(state)` on the minimax variable.

    ```gdscript
    var action_to_do: Array = minimax.action(_board)
    ```

### Structure of the 4 arguments specified above

These functions have to be implemented by the user themselves as it is dependent on the game.

1. `func result(state: Array, action: Array, is_adversary: bool) -> Array:`  
    Should return the resultant state from performing the action.

2. `func terminal(state: Array) -> bool:`  
    Should return `true` if the no further action can take place, otherwise, it should return `false`.

3. `func utility(state: Array, is_adversary: bool) -> float:`  
    Should return the value of the given state. Usually positive for states in which the AI wins and negative for states in which the AI lose.

4. `func possible_actions(state: Array) -> Array[Array]:`  
    Should return all the possible actions that can happen in the given state. Each action is an array item inside the array that is being returned.

Look into the tictactoe demo to gain a better understanding.

## SDQN (Simple Deep Q-Network)

This class implements a Simple Deep Q-Network (SDQN) for reinforcement learning in Godot using the NeuralNetworkAdvanced class.

### How to use SDQN class

1. Initialise the SDQN class with state and action space dimensions:

    ```gdscript
    var sdqn: SDQN = SDQN.new(state_space, action_space, learning_rate)
    ```

2. Set the Q-network for the SDQN:

    ```gdscript
    var neural_network: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()
    --code to change neural network properties--
	sdqn.set_Q_network(neural_network)
    ```

3. Choose an action based on the current state using epsilon-greedy policy:

    ```gdscript
    var action: int = sdqn.choose_action(state)
    ```

4. Add a new experience to the replay memory and train the network:

    ```gdscript
    sdqn.add_memory(state, action, reward, next_state, done)
    ```

5. Save the SDQN model to a file:

    ```gdscript
    sdqn.save("res://path/to/save/file.json")
    ```

6. Load the SDQN model from a file:

    ```gdscript
    sdqn.load("res://path/to/save/file.json")
    ```

### Configurable Parameters in SDQN

1. **Discount Factor**: Default value is `0.95`
    ```gdscript
    sdqn.discount_factor = 0.95  # Or any other float
    ```

2. **Exploration Probability**: Default value is `1`
    ```gdscript
    sdqn.exploration_probability = 1  # Or any other float
    ```

3. **Minimum Exploration Probability**: Default value is `0.01`
    ```gdscript
    sdqn.min_exploration_probability = 0.01  # Or any other float
    ```

4. **Exploration Decay**: Default value is `0.999`
    ```gdscript
    sdqn.exploration_decay = 0.999  # Or any other float
    ```

5. **Batch Size**: Default value is `196`
    ```gdscript
    sdqn.batch_size = 196  # Or any other integer
    ```

    Batch size is the number of elements taken from memory to train on after `max_steps` steps have been done. Each call to `add_memory` is considered one step.

6. **Max Steps**: Default value is `128`
    ```gdscript
    sdqn.max_steps = 128  # Or any other integer
    ```

7. **Target Update Frequency**: Default value is `1024`
    ```gdscript
    sdqn.target_update_frequency = 1024  # Or any other integer
    ```

    Target update frequency is the number of steps after which the target network is replaced with the current network.

8. **Max Memory Size**: Default value is `60 * 60 * 4`
    ```gdscript
    sdqn.max_memory_size = 60 * 60 * 4  # Or any other integer
    ```

9. **Automatic Decay**: Default value is `false`
    ```gdscript
    sdqn.automatic_decay = false  # Or true
    ```

    If `automatic_decay` is enabled, the current exploration probability and learning rate (if decaying) are multiplied with their respective decay values after `max_steps` steps.

10. **Learning Rate Decay Rate**: Default value is `1`
    ```gdscript
    sdqn.lr_decay_rate = 1  # Or any other float
    ```

11. **Final Learning Rate**: Default value is `0.0001`
    ```gdscript
    sdqn.final_learning_rate = 0.0001  # Or any other float
    ```

12. **Use Multi-Threading**: Default value is `false`. Currently not working 100% of the time.
    ```gdscript
    sdqn.use_multi_threading = false  # Or true
    ```


### Methods in SDQN

1. **use_threading()**: Enables multi-threading for training.
    ```gdscript
    sdqn.use_threading()
    ```

2. **set_Q_network(neural_network: NeuralNetworkAdvanced)**: Sets the Q-network for the SDQN.
    ```gdscript
    sdqn.set_Q_network(neural_network)
    ```

3. **set_clip_value(clip_value: float)**: Sets the clip value for the Q-network.
    ```gdscript
    sdqn.set_clip_value(clip_value)
    ```

4. **set_lr_value(lr: float)**: Sets the learning rate for the Q-network.
    ```gdscript
    sdqn.set_lr_value(lr)
    ```

5. **update_lr_linearly()**: Updates the learning rate linearly.
    ```gdscript
    sdqn.update_lr_linearly()
    ```

6. **choose_action(state: Array)**: Chooses an action based on the current state using epsilon-greedy policy.
    ```gdscript
    var action: int = sdqn.choose_action(state)
    ```

7. **add_memory(state: Array, action: int, reward: float, next_state: Array, done: bool)**: Adds a new experience to the replay memory.
    ```gdscript
    sdqn.add_memory(state, action, reward, next_state, done)
    ```

8. **close_threading()**: Closes the multi-threading.
    ```gdscript
    sdqn.close_threading()
    ```

9. **copy()**: Copies the SDQN instance.
    ```gdscript
    var copied_sdqn: SDQN = sdqn.copy()
    ```

10. **to_dict()**: Converts the SDQN instance to a dictionary.
    ```gdscript
    var data: Dictionary = sdqn.to_dict()
    ```

11. **from_dict(dict: Dictionary)**: Loads the SDQN instance from a dictionary.
    ```gdscript
    sdqn.from_dict(dict)
    ```

12. **save(file_path: String)**: Saves the SDQN instance to a file.
    ```gdscript
    sdqn.save("res://path/to/save/file.json")
    ```

13. **load(file_path: String)**: Loads the SDQN instance from a file.
    ```gdscript
    sdqn.load("res://path/to/save/file.json")
    ```

### Note

1. To change the clip value, learning rate, and Q-network, the respective functions (`set_clip_value`, `set_lr_value`, `set_Q_network`) must be called instead of directly setting the variable.
2. To learn more, check out the SDQN Demo available in the Demos folder.