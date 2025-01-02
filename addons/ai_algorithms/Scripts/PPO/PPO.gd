class_name PPO

var policy_network: NeuralNetworkAdvanced
var value_network: NeuralNetworkAdvanced

var state_space: int
var action_space: int

var discount_factor: float = 0.99
var lr_policy = 0.001
var lr_value = 0.001
var clip_ratio = 0.2

var trajectory: Array = []
var max_trajectory_length: int = 5000

enum TRAIN_TYPE {NONE, STEPS, RESETS} # NONE: no training, STEPS: train after a certain number of steps, RESETS: train after a certain number of resets
var train_type: int = PPO.TRAIN_TYPE.NONE
var train_counter: int = 0
var max_train_counter: int = 128

enum TJ_INDEX {STATE, ACTION, REWARD, NEXT_STATE, DONE}

var gae_discount_factor: float = 0.95

func _init(_state_space: int, _action_space: int) -> void:
    state_space = _state_space
    action_space = _action_space

    policy_network = NeuralNetworkAdvanced.new(NeuralNetworkAdvanced.methods.ADAM)
    policy_network.add_layer(state_space)
    policy_network.add_layer(64, "RELU")
    policy_network.add_layer(action_space, "LINEAR")

    value_network = NeuralNetworkAdvanced.new(NeuralNetworkAdvanced.methods.ADAM)
    value_network.add_layer(state_space)
    value_network.add_layer(64, "RELU")
    value_network.add_layer(1, "LINEAR")

func set_hyperparameters(_discount_factor: float, _lr_policy: float, _lr_value: float, _clip_ratio: float) -> void:
    discount_factor = _discount_factor
    lr_policy = _lr_policy
    lr_value = _lr_value
    clip_ratio = _clip_ratio

    policy_network.learning_rate = lr_policy
    value_network.learning_rate = lr_value

func add_trajectory(previous_state: Array, previous_action: Array, reward: float, state: Array, done: bool) -> void:
    assert(previous_state.size() == state_space)
    assert(state.size() == state_space)
    assert(previous_action.size() == action_space)

    trajectory.append([previous_state, previous_action, reward, state, done])
    if trajectory.size() > max_trajectory_length:
        trajectory.pop_front()
    
    train_counter += 1

func calculate_advantages(batch: Array) -> Array[float]:
    return compute_gae(batch)

# Using GAE as an advantage function
func compute_gae(batch: Array) -> Array[float]:
    var T: int = batch.size()
    var advantages: Array[float] = []
    for _void in range(T):
        advantages.append(0.0)
    
    for t in range(T - 1, -1, -1):
        var state_value: float = value_network.predict(batch[t][TJ_INDEX.STATE])[0]
        var next_state_value: float = value_network.predict(batch[t][TJ_INDEX.NEXT_STATE])[0]
        # the TD residual
        var delta: float = batch[t][TJ_INDEX.REWARD] + discount_factor * next_state_value * (1 - batch[t][TJ_INDEX.DONE]) - state_value
        if t == T - 1:
            advantages[t] = delta
        else:
            advantages[t] = delta + discount_factor * gae_discount_factor * advantages[t + 1] * (1 - batch[t][TJ_INDEX.DONE])

    return advantages