#~flow/examples/exp_configs/rl/multiagent/multiagent_figure_eight.py

from copy import deepcopy
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import ContinuousRouter
from flow.controllers import IDMController #the human vehicles follows a physics algo
from flow.controllers import RLController  #the av drives according to RL algo
from flow.controllers import SimLaneChangeController #Controller used to enforce sumo lane-change dynamics on a vehicle.
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams,SumoLaneChangeParams #lane change params
from flow.core.params import VehicleParams
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS #specifies the env structure (for example, the number of lanes)
from flow.envs.multiagent import MultiAgentAccelEnv
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# time horizon of a single rollout
HORIZON    = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 4
# number of parallel workers
N_CPUS     = 2
# number of human-driven vehicles
N_HUMANS   = 10 #increased number of human cars to make it more dramatic
# number of automated vehicles
N_AVS      = 1

    # ContinuousRouter controller -> to perpetually maintain the vehicle within the network.
# lane_change_controller=(SimLaneChangeController, {}) -> used to enforce sumo lane-change dynamics on a vehicle.
# lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic",) - > cars can change lane

vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode="aggressive",),
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=N_HUMANS)

#RL agent
vehicles.add(
    veh_id='rl',
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic",),
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='obey_safe_speed',
    ),
    num_vehicles=N_AVS)

flow_params = dict(
    # name of the experiment
    exp_tag='multiagent_figure_eight',

    # name of the flow environment the experiment is running on
    env_name=MultiAgentAccelEnv,

    # name of the network class the experiment is running on
    network=FigureEightNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step =0.1,
        render   =False,
        restart_instance = True,
        ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'perturb_weight': 0.9, #0.03, #weight of the adversarial agent
            'sort_vehicles': False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy(), 'adversary': gen_policy()}


def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    return agent_id

ADDITIONAL_NET_PARAMS = {
    # radius of the circular components
    "radius_ring": 30,
    # number of lanes
    "lanes": 2,  #CHANGE HERE to add more lanes to the environment
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40
}


# name of the network
name = "simple_training_example_ring_v1"

# network-specific parameters
net_params = NetParams(
        additional_params={
            "radius_ring": 30,
            "lanes": 2,
            "speed_limit": 40,
            "resolution": 40,
        }, )
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
