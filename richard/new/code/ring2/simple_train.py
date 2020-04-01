#
# This versiopn is based on the version in exp_configs
# 
import flow.networks as networks

print(networks.__all__)
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams,SumoLaneChangeParams #lane change params
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs import WaveAttenuationPOEnv
from flow.networks import RingNetwork
from flow.controllers import SimLaneChangeController #Controller used to enforce sumo lane-change dynamics on a vehicle.


# ring road network class
network_name = RingNetwork
# input parameter classes to the network class

# name of the network
name = "simple_training_example_ring2_v1"

# network-specific parameters
net_params = NetParams(
        additional_params={
            "length": 400,
            "lanes": 2,
            "speed_limit": 40,
            "resolution": 40,
        }, )

# initial configuration to vehicles
initial_config = InitialConfig(spacing="uniform", perturbation=1)

# vehicles class

# vehicles dynamics models
# time horizon of a single rollout
HORIZON = 2000
# number of rollouts per training iteration
N_ROLLOUTS = 5
# number of parallel workers
N_CPUS = 2


# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic",),
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=50)
vehicles.add(
    veh_id="rl",
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode="aggressive",),
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='obey_safe_speed',
    ),
    num_vehicles=1)


from flow.core.params import SumoParams
sim_params = SumoParams(sim_step=0.1, render=False)     


# EnvParams specifies environment and experiment-specific parameters that either affect the training process or the dynamics of various components within the network. For the environment WaveAttenuationPOEnv, these parameters are used to dictate bounds on the accelerations of the autonomous vehicles, as well as the range of ring lengths (and accordingly network densities) the agent is trained on.

#Finally, it is important to specify here the horizon of the experiment, which is the duration of one episode (during which the RL-agent acquire data).

from flow.core.params import EnvParams

env_params = EnvParams(
    # length of one rollout
    horizon=HORIZON,
    warmup_steps=750,
    clip_actions=False,

    additional_params={
        # maximum acceleration of autonomous vehicles
        "max_accel": 1,
        # maximum deceleration of autonomous vehicles
        "max_decel": 1,
        # bounds on the ranges of ring road lengths the autonomous vehicle 
        # is trained on
        "ring_length": [490, 520],
    },
)

#Now, we have to specify our Gym Environment and the algorithm that our RL agents will use. Similar to the network, we choose to use on of Flow's builtin environments, a list of which is provided by the script below.

import flow.envs as flowenvs

print(flowenvs.__all__)

#We will use the environment "WaveAttenuationPOEnv", which is used to train autonomous vehicles to attenuate the formation and propagation of waves in a partially observable variable density ring road. To create the Gym Environment, the only necessary parameters are the environment name plus the previously defined variables. These are defined as follows:

from flow.envs import WaveAttenuationPOEnv

env_name = WaveAttenuationPOEnv

# Creating flow_params. Make sure the dictionary keys are as specified. 
flow_params = dict(
    # name of the experiment
    exp_tag=name,
    # name of the flow environment the experiment is running on
    env_name=env_name,
    # name of the network class the experiment uses
    network=network_name,
    # simulator that is used by the experiment
    simulator='traci',
    # simulation-related parameters
    sim=sim_params,
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
    # network-related parameters (see flow.core.params.NetParams and
    # the network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,
    # vehicles to be placed in the network at the start of a rollout 
    # (see flow.core.vehicles.Vehicles)
    veh=vehicles,
    # (optional) parameters affecting the positioning of vehicles upon 
    # initialization/reset (see flow.core.params.InitialConfig)
    initial=initial_config
)


import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


# number of parallel workers
N_CPUS = 2

ray.init(num_cpus=N_CPUS)

# The algorithm or model to train. This may refer to "
#      "the name of a built-on algorithm (e.g. RLLib's DQN "
#      "or PPO), or a user-defined trainable function or "
#      "class registered in the tune registry.")
alg_run = "PPO"

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [32,32,32]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 40  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 5,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": 200,  # number of iterations to stop after
        },
    },
})
