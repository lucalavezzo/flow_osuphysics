import os
import json
import ray
import numpy as np
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.networks import Network
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams, SumoCarFollowingParams, InFlows, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, SimLaneChangeController
from gym.spaces.box import Box
from gym.spaces import Tuple

from constructionRouter import ConstructionRouter
from constructionEnvV2 import myEnv

# time horizon of a single rollout
HORIZON = 5000 #5000
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 10

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
    edges_distribution=["edge4"]
)

# specify the routes for vehicles in the network
class Network(Network):

    def specify_routes(self, net_params):
        return {
                "edge1": ["edge1","edge2","edge3","edge4","edge5","edge6"],
                #"edge3": ["edge3","edge4","edge5","edge10","edge11","edge12","edge3"],
                "edge4": ["edge4","edge5","edge10","edge11","edge12","edge3","edge4"],
                }


inflow = InFlows()
inflow.add(veh_type="human",
           edge="edge1",
           vehs_per_hour=10000,
            depart_lane="random",
            depart_speed="random",
            color="white")

file_dir = "/home/llavezzo/"
net_params = NetParams(
    template="/users/PAS1585/llavez99/work/rl_flow/flow_osuphysics/lucalavezzo/construction/constructionV5.net.xml",
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
        restart_instance=True,
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
            "lane_change_duration": 5,
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
    config["model"].update({"fcnet_hiddens": [32, 32, 32, 32]})
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
