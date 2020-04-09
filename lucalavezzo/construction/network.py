from flow.networks import Network

# specify the routes for vehicles in the network
class Network(Network):

    def specify_routes(self, net_params):
        return {
                "edge1": ["edge1","edge2","edge3","edge4","edge5","edge6"],
                #"edge3": ["edge3","edge4","edge5","edge10","edge11","edge12","edge3"],
                "edge4": ["edge4","edge5","edge10","edge11","edge12","edge3","edge4"],
                }