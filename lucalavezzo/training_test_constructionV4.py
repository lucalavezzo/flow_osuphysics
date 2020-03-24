import json
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
import numpy as np
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from gym.spaces.box import Box
from gym.spaces import Tuple
from flow.core.params import InFlows
from flow.controllers import SimLaneChangeController
from flow.networks import Network
import os
from flow.controllers.routing_controllers import ConstructionRouter
from flow.core.params import SumoLaneChangeParams

from env_constructionV4_padding import myEnv

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 2,
    "max_decel": 2,
}

# time horizon of a single rollout
HORIZON = 5000
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 2

vehicles = VehicleParams()
vehicles.add("rl",
             acceleration_controller=(RLController, {}),
             lane_change_controller=(SimLaneChangeController, {}),
             routing_controller=(ConstructionRouter, {}),
             car_following_params=SumoCarFollowingParams(
                 speed_mode="obey_safe_speed",  
             ),
             num_vehicles=10
             )
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             lane_change_controller=(SimLaneChangeController, {}),
             car_following_params=SumoCarFollowingParams(
                 speed_mode=25
             ),
             lane_change_params = SumoLaneChangeParams(lane_change_mode=1621),
             num_vehicles=0)

# specify the edges vehicles can originate on
initial_config = InitialConfig(
    edges_distribution=["gneE43"]
)

# specify the routes for vehicles in the network
class Network(Network):

    def specify_routes(self, net_params):
        return {
                "gneE43": ["gneE43","gneE4.264","gneE4.264.110","gneE8","gneE9","gneE9.252","gneE33"],
               "gneE8": ["gneE8","gneE9","gneE37","gneE38","gneE39","gneE4.264.110","gneE8"]
               }


inflow = InFlows()
inflow.add(veh_type="human",
           edge="gneE43",
           vehs_per_hour=10000,
            depart_lane="random",
            depart_speed="random",
            color="white")

file_dir = "/home/llavezzo/"
net_params = NetParams(
    template="/mnt/c/Users/llave/Documents/GitHub/flow_osuphysics/lucalavezzo/constructionV4.net.xml",
    inflows=inflow
)

flow_params = dict(
    # name of the experiment
    exp_tag="construction_traffic",

    # name of the flow environment the experiment is running on
    env_name=myEnv,  # <------ here we replace the environment with our new environment

    # name of the network class the experiment is running on
    network=Network,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        restart_instance=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=1000,
        clip_actions=False,
        additional_params={
            "target_velocity": 50,
            "sort_vehicles": False,
            "max_accel": 2,
            "max_decel": 2,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial_config
)


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16, 16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config

alg_run, gym_name, config = setup_exps()
ray.init(num_cpus=N_CPUS + 1)
trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 5,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": 200,
        },
    }
})
