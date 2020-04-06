import random
import numpy as np

from flow.controllers.base_routing_controller import BaseRouter
from flow.controllers.routing_controllers import ContinuousRouter


class ConstructionRouter(ContinuousRouter):

    def choose_route(self, env):

        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge == "edge4":
            new_route = env.available_routes[edge][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route

# class ConstructionRouter(ContinuousRouter):

#     def choose_route(self, env):

#         new_route = env.available_routes["test"][0][0]

#         return new_route
