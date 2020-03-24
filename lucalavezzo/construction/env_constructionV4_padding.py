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
    "max_accel": 2,
    "max_decel": 2,
}

class myEnv(Env):

    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            #shape=(2*self.initial_vehicles.num_vehicles,),
            shape=(2*1000,),
            dtype=np.float32
        )

    def _apply_rl_actions(self, rl_actions):
        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.k.vehicle.get_rl_ids()
        sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(),
                               key=self.k.vehicle.get_x_by_id)

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_actions)

    def get_state(self, **kwargs):
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()
        
        #normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        pos2=[]
        vel2=[]
        for veh_id in ids:
            r = self.k.vehicle.get_x_by_id(veh_id)
            v = self.k.vehicle.get_speed(veh_id)
            edge = self.k.vehicle.get_edge(veh_id)
            
            if type(r) is int or type(r) is float or type(r) is long:
                if -100000 <= r <= 100000:
                    pos2.append(r/max_length)
                else: print("VALUE ERROR VEL: OUTSIDE RANGE", r)      #FIXME: why does it fail? collisions? penalize
            else: print("TYPE ERROR POS", r)
            if type(v) is int or type(v) is float or type(v) is long:
                if -100000 <= v <= 100000:
                    vel2.append(v/max_speed)
                else: print("VALUE ERROR VEL: OUTSIDE RANGE", v)
            else: print("TYPE ERROR VEL", v)

        # the speeds and positions are concatenated to produce the state
        len_zeros = 2000-len(np.concatenate((pos2,vel2)))
        zeros = np.zeros(len_zeros)

        return np.concatenate((pos2,vel2,zeros))

    def compute_reward(self, rl_actions, **kwargs):
        ids = self.k.vehicle.get_ids()
        speeds = self.k.vehicle.get_speed(ids)

        targetSpeeds = []
        for veh_id in ids: 
            edge = self.k.vehicle.get_edge(veh_id)
            if edge == "gneE4.264.110" or edge == "gneE4.264":
                targetSpeeds.append(self.k.vehicle.get_speed(veh_id))

        if(len(targetSpeeds)==0): return 0

        return np.mean(targetSpeeds)
