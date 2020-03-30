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
from flow.envs import Env


ADDITIONAL_ENV_PARAMS = {
    "max_accel": 2,
    "max_decel": 2,
    "lane_change_duration": 5,
}

MAX_EDGE = 12
MAX_LANE = 4
observation_edges = [2,3,4,5]

class myEnv(Env):

    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * num_actions
        ub = [max_accel, 1] * num_actions

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            #shape=(2*self.initial_vehicles.num_vehicles,),
            shape=(5000,),
            dtype=np.float32
        )

    def _apply_rl_actions(self, rl_actions):
        num_rl = self.k.vehicle.num_rl_vehicles
        rl_ids = self.k.vehicle.get_rl_ids()
        sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(),
                               key=self.k.vehicle.get_x_by_id)
        
        acceleration = rl_actions[::2][:num_rl]
        direction = np.round(rl_actions[1::2])[:num_rl]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [
            self.time_counter <= self.env_params.additional_params[
                'lane_change_duration'] + self.k.vehicle.get_last_lc(veh_id)
            for veh_id in sorted_rl_ids]
        
        #FIXME: check that vehicles in lane 4 don't try to change to lane 3
        for veh_id in sorted_rl_ids:
            if(self.k.vehicle.get_edge == "edge4"):
                if(direction[veh_id] == 1): direction[veh_id] = 0

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def get_state(self, **kwargs):
        
        ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        
        #normalizing constants
        max_speed = self.k.network.max_speed()
        for veh_id in ids:
            if(self.k.vehicle.get_speed(veh_id) > max_speed): max_speed = self.k.vehicle.get_speed(veh_id)
        max_length = self.k.network.length()
        
        pos=[]
        vel=[]
        edges=[]
        lanes=[]
        types=[]
        for veh_id in ids:
            
            #RL or human agent
            if(veh_id in rl_ids): types.append(1)
            else: types.append(0)
                
            #Extract edge number
            edge_num = self.k.vehicle.get_edge(veh_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                edge_num = -1
            else:
                edge_num = edge_num[4:]

            #Ignore human vechicles that aren't in the path of the rl agents
            if(veh_id not in rl_ids and edge_num not in observation_edges): continue
                
            r = self.k.vehicle.get_x_by_id(veh_id)
            v = self.k.vehicle.get_speed(veh_id)
            lane_num = self.k.vehicle.get_lane(veh_id)
            
            
            #Append state info
            
            edge_num = int(edge_num)/MAX_EDGE
            lane_num = int(lane_num)/MAX_LANE
            if -1 <= edge_num <= 1:
                edges.append(edge_num)
            else: print("VALUE ERROR EDGE: OUTSIDE RANGE", edge_num)
            if -1 <= lane_num <= 1:
                lanes.append(lane_num)
            else: print("VALUE ERROR LANE: OUTSIDE RANGE", lane_num)
            
            r = r/max_length
            v = v/max_speed
            if type(r) is int or type(r) is float or type(r) is long:
                if -1 <= r <= 1:
                    pos.append(r)
                else: print("VALUE ERROR POS: OUTSIDE RANGE", r)
            else: print("TYPE ERROR POS", r)
            if type(v) is int or type(v) is float or type(v) is long:
                if -1 <= v <= 1:
                    vel.append(v)
                else: print("VALUE ERROR VEL: OUTSIDE RANGE", v)
            else: print("TYPE ERROR VEL", v)
            

        # the speeds and positions are concatenated to produce the state
        if(len(np.concatenate((pos,vel,edges,lanes,types)))>5000): print("STATE EXCEEDS OSBERVATION SIZE")
        len_zeros = 5000-len(np.concatenate((pos,vel,edges,lanes,types)))
        zeros = np.zeros(len_zeros)

        return np.concatenate((pos,vel,edges,lanes,types,zeros))

    def compute_reward(self, rl_actions, **kwargs):
        ids = self.k.vehicle.get_ids()
        speeds = self.k.vehicle.get_speed(ids)

        #Only count speeds of cars in edge prior to the 'construction site'
        targetSpeeds = []
        for veh_id in ids: 
            edge = self.k.vehicle.get_edge(veh_id)
            if edge == "gneE4.264.110" or edge == "gneE4.264":
                targetSpeeds.append(self.k.vehicle.get_speed(veh_id))

        if(len(targetSpeeds)==0): return 0

        return np.mean(targetSpeeds)
