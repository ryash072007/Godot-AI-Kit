---
<p>
Project will continue once I complete CS50ai.

TODO:
1. Multi-Layer Neural Network Support (more than a single hidden network)
2. PPO Support (Unrealistic but will try!)
</p>
---

<h1 id="ai-algorithm-for-godot-4">AI Algorithm for Godot 4</h1>
<p>The goal of this project is to provide a variety of AI Algorithms in Godot 4 natively using GDscript.</p>
<h2 id="index">Index</h2>
<ol>
<li><a href="#simple-neural-network-and-neural-net-plugin-for-godot-4">Simple Neural Network and Neural Net</a></li>
<li><a href="#q-learning-algorithm">Q-Learning Algorithm</a></li>
</ol>
<h2 id="simple-neural-network-and-neural-net-plugin-for-godot-4">Simple Neural Network and Neural Net Plugin for Godot 4</h2>
<p>This part of the plugin allows you to create a Multi Layer Neural Network and also provides a NeuralNet by which you can easily automatically train the network (which can be found under Node2D Section in the add node window).<br>
This plugin is intended for creating AIs that can complete a game level.</p>
<h3 id="rules-to-be-followed-if-using-neural-net">Rules to be followed if using Neural Net</h3>
<ol>
<li>If using Neural Net, the identifier or name of the variable of the Neural Network used in your code has to be <code>nn</code>. Like this:</li>
</ol>
<pre><code>var nn: NeuralNetwork
</code></pre>
<p>This is because the Neural Net only works when the Neural Network is named as <code>nn</code>.</p>
<ol start="2">
<li>If using Neural Net, make sure you do not assign your Neural Network Variable <code>nn</code> anything. All you are supposed to do is declare it like this:</li>
</ol>
<pre><code>var nn: NeuralNetwork
</code></pre>
<p>This is because the Neural Net depends on the fact that <code>nn</code> is not assigned anything.</p>
<ol start="3">
<li>When your AI or player has to be killed or removed, always use the <code>queue_free()</code> method. This is because the Neural Net relies on the signal emitted by the node when exiting the tree to recieve the fitness and Neural Network of that node. Example:</li>
</ol>
<pre><code>Object.queue_free()
</code></pre>
<h3 id="what-each-variable-means-and-how-to-use-them">What each variable means and how to use them</h3>
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
<h3 id="how-to-use-neural-net">How to use Neural Net</h3>
<p>Just ensure that all the variables/properties mentioned above are correctly set. The position of this node is where all the AIs will be spawned, meaning, the position of this node = position of AI when spawned.</p>
<h3 id="how-to-use-neural-network">How to use Neural Network</h3>
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
<ol start="8">
<li>You can use the <code>predict(input_array: Array[float])</code> function also to get predictions. Example:</li>
</ol>
<pre><code>var output = nn.predict([0.0, 6, 0.2])
</code></pre>
<ol start="9">
<li>If you know the expected output of an input, you can use the <code>train(input_array: Array, target_array: Array)</code> function in a loop. Example:</li>
</ol>
<pre><code>for epoch in range(2000):
    nn.train([0, 1], [1])
    nn.train([1, 1], [1])
    nn.train([0, 0], [0])
    nn.train([1, 1], [0])
</code></pre>
<ol start="10">
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
<h2 id="q-learning-algorithm">Q-Learning Algorithm</h2>
<p>This algorithm implements Q-Learning algorithm using Q-Table natively in Godot.</p>
<h3 id="how-to-use-qlearning-class">How to use QLearning class</h3>
<ol>
<li>
<p>Initialise a QLearning variable</p>
<pre><code>var qnet: QLearning = QLearning.new(observation_space, action_space)
</code></pre>
<p>Both the <code>observation_space</code> and <code>action_space</code> have to be natural numbers representing the possible states the agent can be in and the possible actions choices the agent can take in any given state.</p>
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
<h3 id="configurable-values">Configurable Values</h3>
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
<h3 id="things-to-keep-in-mind-when-using-qlearning">Things to keep in mind when using QLearning</h3>
<ol>
<li>The predict method of the QLearning class takes two compulsory arguments:<pre><code>qnet.predict(current_state, previous_state_reward)
</code></pre>
The <code>current_state</code> has to be a whole number representing the state it is currently in, while the <code>previous_state_reward</code> has to a float representing the reward it got for the previous action it took.</li>
</ol>
<h2 id="credits">Credits</h2>
<p>NeuralNet basis: Greaby</p>

