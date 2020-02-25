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

# import the base environment class
from flow.envs import Env

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 1,
    "max_decel": 1,
}

class myEnv(Env):

    @property
    def action_space(self):
        num_actions = 1
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

    @property
    def observation_space(self):
        return Box(
            low=float(-1000000000),
            high=float("inf"),
            shape=(2*self.k.vehicle.num_vehicles,),
        )

    def _apply_rl_actions(self, rl_actions):
        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.k.vehicle.get_rl_ids()

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)

    def get_state(self, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()

        # we use the get_absolute_position method to get the positions of all vehicles
        pos = [self.k.vehicle.get_x_by_id(veh_id) for veh_id in ids]

        # we use the get_speed method to get the velocities of all vehicles
        vel = [self.k.vehicle.get_speed(veh_id) for veh_id in ids]

        # the speeds and positions are concatenated to produce the state
        return np.concatenate((pos, vel))

    def compute_reward(self, rl_actions, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()

        # we next get a list of the speeds of all vehicles in the network
        speeds = self.k.vehicle.get_speed(ids)

        # finally, we return the average of all these speeds as the reward
        return np.mean(speeds)
