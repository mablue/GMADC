import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab
class CallGraph:

    def __init__(self,csrgraph ):
        self.csrgraph = csrgraph
        super().__init__()

    def draw(self):

        G = nx.from_scipy_sparse_matrix(self.csrgraph)

        # G.add_edges_from([('A', 'B'),('C','D'),('G','D')], weight=1)
        # G.add_edges_from([('D','A'),('D','E'),('B','D'),('D','E')], weight=2)
        # G.add_edges_from([('B','C'),('E','F')], weight=3)
        # G.add_edges_from([('C','F')], weight=4)
        # G.from_scipy_sparse_matrix(self.csrgraph)


        val_map = {'A': 1.0,
                        'D': 0.5714285714285714,
                                    'H': 0.0}

        values = [val_map.get(node, 0.45) for node in G.nodes()]
        # edge_labels=dict([((u,v,),d[1])
        #                 for u,v,d in G.edges(data=True)])

        # red_edges = [('C','D'),('D','A')]
        # edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

        pos=nx.spring_layout(G)
        # nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw_networkx_labels(G, pos)
        nx.draw(G,pos, node_size=1,edge_cmap=plt.cm.Reds)
        pylab.show()
