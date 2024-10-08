
TODO:

~~1. Multi-Layer Neural Network Support (more than a single hidden network)~~ [Completed]

~~2. PPO Support (Unrealistic but will try!)~~ [On Hold]

3. Simple DQN Support [will start soon]

</p>


<h1  id="ai-algorithm-for-godot-4">AI Algorithm for Godot 4</h1>

<p>The goal of this project is to provide a variety of AI Algorithms in Godot 4 natively using GDscript.</p>

<h2  id="index">Index</h2>

<ol>

<li><a  href="#simple-neural-network-and-neural-net-plugin-for-godot-4">Simple Neural Network and Neural Net</a></li>

<li><a  href="#NNA">Neural Network Advanced (Multi-Layered Neural Network)</a></li>

<li><a  href="#q-learning-algorithm">Q-Learning Algorithm (and SARSA)</a></li>

<li><a  href="#minimax-algorithm">Minimax Algorithm</a></li>

</ol>

<h2  id="simple-neural-network-and-neural-net-plugin-for-godot-4">Simple Neural Network and Neural Net Plugin for Godot 4</h2>

<p>This part of the plugin allows you to create a Multi Layer Neural Network and also provides a NeuralNet by which you can easily automatically train the network (which can be found under Node2D Section in the add node window).<br>

This plugin is intended for creating AIs that can complete a game level.</p>

<h3  id="rules-to-be-followed-if-using-neural-net">Rules to be followed if using Neural Net</h3>

<ol>

<li>If using Neural Net, the identifier or name of the variable of the Neural Network used in your code has to be <code>nn</code>. Like this:</li>

</ol>

<pre><code>var nn: NeuralNetwork

</code></pre>

<p>This is because the Neural Net only works when the Neural Network is named as <code>nn</code>.</p>

<ol  start="2">

<li>If using Neural Net, make sure you do not assign your Neural Network Variable <code>nn</code> anything. All you are supposed to do is declare it like this:</li>

</ol>

<pre><code>var nn: NeuralNetwork

</code></pre>

<p>This is because the Neural Net depends on the fact that <code>nn</code> is not assigned anything.</p>

<ol  start="3">

<li>When your AI or player has to be killed or removed, always use the <code>queue_free()</code> method. This is because the Neural Net relies on the signal emitted by the node when exiting the tree to recieve the fitness and Neural Network of that node. Example:</li>

</ol>

<pre><code>Object.queue_free()

</code></pre>

<h3  id="what-each-variable-means-and-how-to-use-them">What each variable means and how to use them</h3>

<ol>

<li>Ai Scene: This is where you will assign the AI or Player scene by clicking on the drop down arrow on the right side, clicking <code>quick load</code> and selecting your scene.</li>

<li>Batch Size: This is the informal Batch Size of each generation. The actual batch size of each generation is emitted by the <code>true_batch_size</code> signal. This controls the base amount of AIs spawned.</li>

<li>Generation Delay: This is the time limit (in seconds) for any generation. Once a generation has lived longer than the amount specified in this, the generation is reset and the next generation comes.</li>

<li>Input Nodes: This is where the input nodes for the <code>nn</code> will be set. Input Nodes means how many different inputs will the <code>nn</code> recieve.</li>

<li>Hidden Nodes: This is where the hidden nodes for the <code>nn</code> will be set. Hidden Nodes means how many nodes will process the data given by the input nodes. You should experiment with this amount.</li>

<li>Output Nodes: This is where you will set how many outputs you want to recieve by the <code>nn</code>.</li>

<li>Random Population: This determines how many AIs with random <code>nn</code> will be spawned after the first generation (after the 0 generation). It is a good idea to set this to a value greater than 10 as it allows for more possibilites to be explored by the Neural Net.</li>

<li>Use Reproduction: This determines whether reproduction will also be used to create new AIs for the next generations. This enables for combination of different traits of different <code>nn</code>s. However, you will most probably not need this as Random and Mutated Population will suffice.</li>

<li>Reproduced Population: If “Use Reproduction” is checked, this will determine how many AIs will be spawned with reproduced <code>nn</code>s. Note: This value must always be greater than half of the value of Batch Size if you have checked “Use Reproduction” as true.</li>

</ol>

<h3  id="how-to-use-neural-net">How to use Neural Net</h3>

<p>Just ensure that all the variables/properties mentioned above are correctly set. The position of this node is where all the AIs will be spawned, meaning, the position of this node = position of AI when spawned.</p>

<h3  id="how-to-use-neural-network">How to use Neural Network</h3>

<pre><code>var nn: NeuralNetwork = NeuralNetwork.new(input_nodes, hidden_nodes, output_nodes)

</code></pre>

<ol>

<li>

<p>Input Nodes: This is where the input nodes for the <code>nn</code> will be set. Input Nodes means how many different inputs will the <code>nn</code> recieve.</p>

</li>

<li>

<p>Hidden Nodes: This is where the hidden nodes for the <code>nn</code> will be set. Hidden Nodes means how many nodes will process the data given by the input nodes. You should experiment with this amount.</p>

</li>

<li>

<p>Output Nodes: This is where you will set how many outputs you want to recieve by the <code>nn</code>.</p>

</li>

<li>

<p>If the Neural Network depends mostly on inputs from raycasts, you can use the “get_prediction_from_raycasts(optional_val: Array = [])”. This function returns an array of floats which are the outputs. The “optional_val” is optional can be used to give more custom inputs in addition to the raycasts. Example:</p>

</li>

</ol>

<pre><code>var output = nn.get_prediction_from_raycasts()

# or

var output = nn.get_prediction_from_raycasts([0, 0.4, 2])

</code></pre>

<ol  start="8">

<li>You can use the <code>predict(input_array: Array[float])</code> function also to get predictions. Example:</li>

</ol>

<pre><code>var output = nn.predict([0.0, 6, 0.2])

</code></pre>

<ol  start="9">

<li>If you know the expected output of an input, you can use the <code>train(input_array: Array, target_array: Array)</code> function in a loop. Example:</li>

</ol>

<pre><code>for epoch in range(2000):

nn.train([0, 1], [1])

nn.train([1, 1], [1])

nn.train([0, 0], [0])

nn.train([1, 1], [0])

</code></pre>

<ol  start="10">

<li>If you want to mutate your Neural Network, you can do so by:</li>

</ol>

<pre><code>nn = NeuralNetwork.mutate(nn)

</code></pre>

<p>where <code>nn</code> is your Neural Network.<br>

11. If you want to mutate your Neural Network, you can do so by:</p>

<pre><code>new_nn = NeuralNetwork.copy(nn)

</code></pre>

<p>where <code>nn</code> is your Neural Network and <code>new_nn</code> is the new one to which you are copying the <code>nn</code> to.<br>

12. IF you want to reproduce your Neural Network with another, you can do so by:</p>

<pre><code>reproduced_nn = NeuralNetwork.reproduce(nn_1, nn_2)

</code></pre>

<p>where <code>nn_1</code> and <code>nn_2</code> are the parent Neural Networks.</p>

<h2 id="NNA">Neural Network Advanced </h2>
<p> <b>Note:</b> Support for this in the Neural Net has not been implemented yet.</p>
<h3> How to use NeuralNetworkAdvanced class</h3>
<ol>
<li> Initialising the NNA variable

	var nnas: NeuralNetworkAdvanced = NeuralNetworkAdvanced.new()
</li>
<li>Add the first layer to the network. Here you should only specify the number of nodes needed in this layer.

	nnas.add_layer(2)
  </li>
<li> Add the remaining layers to the network. Here you can also specify which activation function to use. Eg:

	nnas.add_layer(4, nnas.ACTIVATIONS.ARCTAN)
	nnas.add_layer(6, nnas.ACTIVATIONS.ARCTAN)
	nnas.add_layer(1, nnas.ACTIVATIONS.SIGMOID)
</li>
<li> To train the network, you can call the <code>train()</code> function on the NNA. The first argument has to be the <b>input array of size same as that of the first layer</b> and the <b>second argument has to be the output array of size same as the last layer of the network.</b>
Note: This only runs a single train call. You need to do a lot of these to train your NNA to be accurate. Eg: Training for an XOR Gate. In the demo, you can see that this code is placed in the <code>_physics_process</code> function so that it is ran many times a second.

	nnas.train([0,0], [0])
	nnas.train([1,0], [1])
	nnas.train([0,1], [1])
	nnas.train([1,1], [0])
</li>
<li> To get a prediction/output from the NNA. You have to call the <code>predict</code> function on the NNA. The first and only argument has to be input  array for the network. It will return an array of the same size as that of the last/output layer. Eg:

	print(nnas.predict([1,0]))
will return <code>[1]</code> when trained.
</ol>
<h3>Note</h3>
<ol>
<li>
Addition of layers should only happen once and so <code>_ready()</code> is an appropriate place to put them.
</li>
<li>
All the possible activations functions can be seen here: [NeuralNetworkAdvanced.gd](https://github.com/ryash072007/Godot-AI-Kit/blob/main/addons/ai_algorithms/Scripts/Neural/Neural_Network_Advanced.gd) under the <code>ACTIVATION</code> dictionary.
</li>
</ol>
<h2  id="q-learning-algorithm">Q-Learning Algorithm</h2>

<p>This algorithm implements Q-Learning algorithm using Q-Table natively in Godot.</p>

<h3  id="how-to-use-qlearning-class">How to use QLearning class</h3>

<ol>

<li>

<p>Initialise a QLearning variable</p>

<pre><code>var qnet: QLearning = QLearning.new(observation_space, action_space, is_learning, not_sarsa)

</code></pre>

<p>Both the <code>observation_space</code> and <code>action_space</code> have to be natural numbers representing the possible states the agent can be in and the possible actions choices the agent can take in any given state. <code>is_learning</code> is a boolean value of whether the agent should be learning or not, and <code>not_sarsa</code> set to <code>true</code> will disable sarsa (on-policy). I would recommend sarsa if you want a safer route to the final path.</p>

</li>

<li>

<p>Get a prediction from the QLearning variable:</p>

<pre><code>qnet.predict(current_state, reward_of_previous_state)

</code></pre>

<p>The above method returns an whole number that lies between <code>0</code> and <code>action_space - 1</code>. The value returned corresponds to an action the agent can take.<br>

You can assign the returned value to variable by:</p>

<pre><code>var action_to_do: int = qnet.predict(current_state, previous_reward)

</code></pre>

</li>

</ol>

<h3  id="configurable-values">Configurable Values</h3>

<ol>

<li>

<p><code>qnet.exploration_probability</code> -&gt; has to be a float value<br>

<mark>Default Value: <code>1.0</code></mark><br>

The probability that the agent will take a random action or exploit the data it has learned.<br>

Do not change unless you know what you are doing.</p>

</li>

<li>

<p><code>qnet.exploration_decreasing_decay</code> -&gt; has to be a float value<br>

<mark>Default Value: <code>0.01</code></mark><br>

Changes how the value by which the <code>qnet.exploration_probability</code> decreases every ```qnet.decay_per_steps`` steps.</p>

</li>

<li>

<p><code>qnet.min_exploration_probability</code> -&gt; has to be a float value<br>

<mark>Default Value: <code>0.01</code></mark><br>

The minimum value the <code>exploration_probability</code> can take.</p>

</li>

<li>

<p><code>qnet.learning_rate</code> -&gt; has to be a float<br>

<mark>Default Value:<code>0.2</code></mark><br>

The rate at which the agent learns.</p>

</li>

<li>

<p><code>qnet.decay_per_steps</code> -&gt; has to be natural number<br>

<mark>Default Value: <code>100</code></mark><br>

After how many steps does the <code>qnet.exploration_probability</code> decrease by <code>qnet.exploration_decreasing_decay</code> value.</p>

</li>

<li>

<p><code>qnet.is_learning</code> -&gt; has to be a bool value<br>

<mark>Default Value: <code>true</code></mark><br>

To be set to false only when the <code>qnet.QTable.data</code> is set manually.</p>

</li>

<li>

<p><code>print_debug_info</code> -&gt; has to be a bool value<br>

<mark>Default Value: <code>false</code></mark><br>

This can be set to <code>true</code> if you want to print debug info - total steps completed and current exploration probability - every <code>qnet.decay_per_steps</code>.</p>

</li>

</ol>

<h3  id="things-to-keep-in-mind-when-using-qlearning">Things to keep in mind when using QLearning</h3>

<ol>

<li>The predict method of the QLearning class takes two compulsory arguments:<pre><code>qnet.predict(current_state, previous_state_reward)

</code></pre>

The <code>current_state</code> has to be a whole number representing the state it is currently in, while the <code>previous_state_reward</code> has to a float representing the reward it got for the previous action it took.</li>

</ol>
<h2  id="minimax-algorithm">Minimax Algorithm</h2>
Alpha-Beta Pruning has been implemented!
<br>
If the AI is playing the role of the adversary, then set <code>minimax.is_adversary</code> to <code>true</code> else <code> false</code>.
<h3  id="how-to-use-minimax-class">How to use Minimax class</h3>
<ol>
<li> Initialise the Minimax class with 4 arguments:
<br>
<code>var  minimax: Minimax  =  Minimax.new(Callable(result),
Callable(terminal),
Callable(utility),
Callable(possible_actions))
</code>
<ol>
<li><code>result_func: Callable</code>: This callable argument must link to the function in your code that returns the state of the environment after a particular action is performed.
</li>
<li> <code>terminal_func: Callable</code>: This callable argument must link to the function in your code that returns <code>true</code> if the game is over and <code>false</code> if the game can continue for a given state.
</li>
<li><code>utility_func: Callable</code>: This callable argument must link to the function in your code that returns the value of the given state. Currently this function only runs when the game is a terminal state. Losing states should have lesser value than winning states.
</li>
<li><code>possible_actions_func: Callable</code>: This callable argument must link to the function in your code that returns all the possible actions for a given state.
</li>
</li>
<li> Every time the AI needs to perform an action, call the <code>action(state)</code> on the minimax variable.<br>
<code>var  action_to_do: Array  =  minimax.action(_board)</code>
</li>
</ol>
<h3  id="structure-arguments-minimax">Structure of the 4 arguments specified above</h3>
These functions have to be implemented by the user themselves as it is dependent on the game.
<ol>
<li>
<code>func  result(state: Array, action: Array, is_adversary: bool) ->  Array:</code><br>
Should return the resultant state from performing the action.
</li>
<li>
<code>func  terminal(state: Array) ->  bool:</code><br>
Should return <code>true</code> if the no further action can take place, otherwise, it should return <code>false</code>.
</li>
<li><code>func  utility(state: Array, is_adversary: bool) ->  float:</code><br>
Should return the value of the given state. Usually positive for states in which the AI wins and negative for states in which the AI lose.
</li>
<li>
<code>func  possible_actions(state: Array) ->  Array[Array]:</code><br>
Should return all the possible actions that can happen in the given state. Each action is an array item inside the array that is being returned.
</li>
</ol>
Look into the tictactoe demo to gain a better understanding.
