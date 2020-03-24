import random
import numpy as np

from flow.controllers.routing_controllers import ContinuousRouter

class ConstructionRouter(ContinuousRouter):
	def choose_route(self,env):
		edge = env.k.vehicle.get_edge(self.veh_id)
		if edge == "gneE8":
			new_route = env.available_routes["gneE8"][0][0]
		else:
			new_route = super().choose_route(env)
		return new_route