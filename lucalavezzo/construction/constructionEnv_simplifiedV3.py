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
from flow.core import rewards

MAX_EDGE = 12
MAX_LANE = 2
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
            shape=(10*8+4,),
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
        
        #Check that vehicles in junction begore edge with
        # 2 lanes don't try to change to lane 3
        for i, veh_id in enumerate(sorted_rl_ids):
            edge_num = self.k.vehicle.get_edge(veh_id)
            if(edge_num == '' or edge_num[0] == ':'):
                direction[i] = 0
            if(edge_num == ':gneJ6'):
                direction[i] = 0

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
            if(abs(self.k.vehicle.get_speed(veh_id)) > 10000): continue
            if(self.k.vehicle.get_speed(veh_id) > max_speed): max_speed = self.k.vehicle.get_speed(veh_id)
        max_length = self.k.network.length()
        
        #rl vehicle info (4*num_rl)
        pos=[]  
        vel=[]  
        edges=[] 
        lanes=[] 

        #follower/leader info (size 6*num_rl)
        types=[]
        follower_dv = []
        follower_dx = []
        leader_dv = []
        leader_dx = []

        #network info (2*2)
        lane_traffic=[0,0] 
        lane_traffic_speed=[0,0]


        #Density and average speed of vehicles in lane before construction zone
        for veh_id in ids:

            edge_num = self.k.vehicle.get_edge(veh_id)
            lane_num = int(self.k.vehicle.get_lane(veh_id))
            v = self.k.vehicle.get_speed(veh_id)
            if(-1 <= v/max_speed <= 1):
                if(edge_num == "edge3" or edge_num == "edge2"):
                    lane_traffic[lane_num] += 1
                    lane_traffic_speed[lane_num] += v/max_speed
                

        #Info for each RL vehicle
        for rl_id in rl_ids:
            
            edge_num = self.k.vehicle.get_edge(rl_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                edge_num = -1
            else:
                edge_num = edge_num[4:]
            lane_num = self.k.vehicle.get_lane(rl_id)
            r = self.k.vehicle.get_x_by_id(rl_id)
            v = self.k.vehicle.get_speed(rl_id)
            
            follower_id = self.k.vehicle.get_follower(rl_id)
            leader_id = self.k.vehicle.get_leader(rl_id)

            if leader_id in ["", None]:
                leader_dv.append(1)
                leader_dx.append(1)
                types.append(0)
            else:
                leader_pos = (self.k.vehicle.get_x_by_id(leader_id) - self.k.vehicle.get_x_by_id(rl_id))/max_length
                leader_speed = self.k.vehicle.get_speed(leader_id)/max_speed

                if -1 <= leader_speed <= 1:
                    leader_dv.append(leader_speed)
                else: 
                    #print("VALUE ERROR LEADER_DV: OUTSIDE RANGE", self.k.vehicle.get_speed(leader_id), leader_id)
                    leader_dv.append(0)

                if -1 <= leader_pos <= 1:
                    leader_dx.append(leader_pos)
                else: 
                    #print("VALUE ERROR LEADER_DX: OUTSIDE RANGE", leader_speed, leader_id)
                    leader_dx.append(0)

                if(rl_id in rl_ids): types.append(1)
                else: types.append(-1)

            if follower_id in ["",None]:
                follower_dv.append(1)
                follower_dx.append(1)
                types.append(0)
            else:
                follower_pos = (self.k.vehicle.get_x_by_id(rl_id) - self.k.vehicle.get_x_by_id(follower_id))/max_length
                follower_speed = self.k.vehicle.get_speed(follower_id)/max_speed

                if -1 <= follower_speed <= 1:
                    follower_dv.append(follower_speed)
                else: 
                    #print("VALUE ERROR FOLLOWER_DV: OUTSIDE RANGE", self.k.vehicle.get_speed(follower_id), follower_id)
                    follower_dv.append(0)
                 
                if -1 <= follower_pos <= 1:
                    follower_dx.append(follower_pos)
                else: 
                    #print("VALUE ERROR FOLLOWER_DV: OUTSIDE RANGE", self.k.vehicle.get_speed(follower_id), follower_id)
                    follower_dx.append(0)

                if(rl_id in rl_ids): types.append(1)
                else: types.append(-1)


            edge_num = int(edge_num)/MAX_EDGE
            lane_num = int(lane_num)/MAX_LANE
            if -1 <= edge_num <= 1:
                edges.append(edge_num)
            else: 
                #print("VALUE ERROR EDGE: OUTSIDE RANGE", edge_num)
                edge.append(0)
            if -1 <= lane_num <= 1:
                lanes.append(lane_num)
            else: 
                #print("VALUE ERROR LANE: OUTSIDE RANGE", lane_num)
                lanes.append(0)

            r = r/max_length
            v = v/max_speed
            
            if -1 <= r <= 1:
                pos.append(r)
            else:
                pos.append(0)

            if -1 <= v <= 1:
                vel.append(v)
            else:
                vel.append(0)


        for i in range(2):
            if(lane_traffic[i] != 0): lane_traffic_speed[i] = (lane_traffic_speed[i]/lane_traffic[i])
            else: lane_traffic_speed[i] = 0
            if(len(ids)!=0): lane_traffic[i] = lane_traffic[i] / len(ids)
            else: lane_traffic[i] = 0

        state = np.concatenate((pos,vel,edges,lanes,
                                types,follower_dx,follower_dv,leader_dv,leader_dx,
                                lane_traffic,lane_traffic_speed))
        
        #if(np.max(state) > 1 or np.min(state) < -1 or len(state) != 108): print("ERROR IN STATE")
            
        return state

    def compute_reward(self, rl_actions, **kwargs):
        return rewards.desired_velocity(self, edge_list = ["edge2","edge3"])
