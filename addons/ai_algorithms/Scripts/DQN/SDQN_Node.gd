extends Node
## This is the script for the SDQN node. It is a wrapper for the SDQN class, which is a deep Q-learning algorithm implementation.

@export_group("Basic Parameters")
## The number of states in the environment.
@export var state_space: int

## The number of actions in the environment.
@export var action_space: int
## Whether to continue training the model.
@export var continue_training: bool = true

@export_group("Load File")
## The file path to load the model from. Leave empty if you do not want to load a model. Save the model using the save function.
@export_file var load_file: String = ""

@export_group("Hyperparameters")
## The learning rate of the model.
@export_range(0,1, 0.00001) var learning_rate: float = 0.001
## The discount factor of the model. Essentially, how much the model should value future rewards.
@export_range(0,1,0.01) var discount_factor: float = 0.95

@export_group("Training Parameters")
## The batch size for training the model every max steps.
@export var batch_size: int = 196
## The maximum number of steps to take in the environment before training. Each add_memory is considered a step.
@export var max_steps: int = 128
## The number of steps before updating the target network.
@export var target_update_frequency: int = 1024
## The maximum size of the replay memory.
@export var max_memory_size: int = 60 * 60 * 4

@export_group("Exploration Parameters")
## The probability that the agent should explore the env instead of exploiting.
@export_range(0,1,0.0001) var exploration_probability: float = 1
## The minimum exploration probability.
@export_range(0,1,0.0001) var min_exploration_probability: float = 0.01
## The rate at which the exploration probability decays. This is multiplied with the exploration probability every max_steps.
@export_range(0,1,0.0001) var exploration_decay: float = 0.999

@export_group("Learning Parameters")
## To automatically decay the learning rate.
@export var automatic_decay: bool = false
## The decay rate of the learning rate. Set it to 1 if you do not want it to decay. This is multiplied with the learning rate every max_steps.
@export_range(0,1,0.0001) var lr_decay_rate: float = 1
## The final learning rate.
@export_range(0,1,0.0001) var final_learning_rate: float = 0.0001


var sdqn: SDQN

func _ready() -> void:
	sdqn = SDQN.new(state_space, action_space)
	if load_file != "":
		sdqn.load(load_file)
		_update_from_sdqn()
		return
	sdqn.discount_factor = discount_factor
	sdqn.exploration_probability = exploration_probability
	sdqn.min_exploration_probability = min_exploration_probability
	sdqn.exploration_decay = exploration_decay
	sdqn.batch_size = batch_size
	sdqn.max_steps = max_steps
	sdqn.target_update_frequency = target_update_frequency
	sdqn.max_memory_size = max_memory_size
	sdqn.automatic_decay = automatic_decay
	sdqn.lr_decay_rate = lr_decay_rate
	sdqn.final_learning_rate = final_learning_rate


func choose_action(state: Array) -> int:
	return sdqn.choose_action(state)

func add_memory(state: Array, action: int, reward: float, next_state: Array, done: bool) -> void:
	if continue_training:
		sdqn.add_memory(state, action, reward, next_state, done)
	_update_from_sdqn()

func save(file_path: String) -> void:
	sdqn.save(file_path)

func set_Q_network(network: NeuralNetworkAdvanced) -> void:
	if load_file != "":
		return
	sdqn.set_Q_network(network)
	sdqn.set_lr_value(learning_rate)

func set_clip_value(value: float) -> void:
	sdqn.set_clip_value(value)

func _update_from_sdqn() -> void:
	exploration_probability = sdqn.exploration_probability
	learning_rate = sdqn.learning_rate
