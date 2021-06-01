"""Environment Class."""
import networkx as nx
import numpy as np
import math
import sys
import pdb
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def dist2dPoints(xa, ya, xb, yb):
    """Euclidean distance between two points (2D)."""
    dx = xb - xa
    dy = yb - ya
    d = math.sqrt(dx**2 + dy**2)
    return d


class Environment(object):
    """Represents the environment using the NetworkX package."""

    def __init__(self, nodes):
        """Instantiate the environment."""
        self.num_nodes = len(nodes)

        sys.stdout.write("Creating the environment")
        self.g = nx.Graph()
        for k in range(self.num_nodes):
            self.g.add_node(k,
                            position=(nodes[k]['x'], nodes[k]['y']),
                            reward=nodes[k]['r'],
                            visited_by=[-1])

        self.distance_matrix = self.computeDistanceMatrix()
        self.unvisited_vertices = set(np.asarray(self.g.nodes))
        self.visited_vertices = set()
        sys.stdout.write(' ' + u'\u2713' + '\n')

    def getNodes(self, nodes=None):
        """Return the set of nodes."""
        nodes_vec = np.asarray(self.g.nodes)
        if nodes == 'all':
            return nodes_vec
        if nodes is None:
            return nodes_vec[1:-1]
        return []

    def getReward(self, id):
        return self.g.nodes[id]['reward']

    def dist2dVertices(self, a, b):
        """Euclidean distance between two vertices of the environment."""
        return dist2dPoints(self.g.nodes[a]['position'][1], self.g.nodes[a]['position'][0],
                            self.g.nodes[b]['position'][1], self.g.nodes[b]['position'][0])

    def computeDistanceMatrix(self):
        """Create and calculate the distance matrix between all nodes in the environment."""
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for x in self.getNodes('all'):
            for y in self.getNodes('all'):
                matrix[x, y] = self.dist2dVertices(x, y)

        return matrix

    def computeVertexUtility(self, vertex, visited):
        alpha = -np.log(0.01) / 2
        # some values were considered constant here but in future work they must be dependent on the agents

        if vertex == 0 or vertex == self.num_nodes-1:
            return 0.0

        extra = 0.0
        reward = 0.0
        for w in self.getNeighborhood(vertex, 2):
            if w != vertex and w not in visited and w != 0 and w != self.num_nodes-1:
                extra += np.exp(-alpha * self.distance_matrix[vertex, w]) * self.getReward(w)

        if vertex in visited:
            reward += self.getReward(vertex)

        return reward+extra

    def computeVertexDisponibility(self, vertex, visited):
        alpha = -np.log(0.01) / 5
        # some values were considered constant here but in future work they must be dependent on the agents

        if vertex == 0 or vertex == self.num_nodes-1:
            return 0.0

        disp = 0.0
        if vertex not in visited:
            disp += self.getReward(vertex)
        else:
            return 0.0

        for w in self.getNeighborhood(vertex, 5):
            if w != vertex and w not in visited and w != 0 and w != self.num_nodes-1:
                disp += np.exp(-alpha * self.distance_matrix[vertex, w]) * self.getReward(w)

        return disp*disp*disp

    def getVerticesDistance(self, x, y):
        return self.distance_matrix[x, y]

    def makePathEdges(self, path):
        edges = []
        a = path[0]
        for b in path[1:]:
            if a != b:
                edges.append((a, b))
            a = b

        return edges

    def visit(self, vertex, agent):
        if vertex in self.unvisited_vertices:
            self.unvisited_vertices.remove(vertex)
            self.visited_vertices.add(vertex)
            self.g.nodes[vertex]['visited_by'][0] = agent
        else:
            if vertex != 0 and vertex != self.num_nodes-1:
                pdb.set_trace()
            self.g.nodes[vertex]['visited_by'].append(agent)

    def unvisit(self, vertex, agent_id):
        if vertex in self.visited_vertices:
            if len(self.g.nodes[vertex]['visited_by']) > 1:
                # pdb.set_trace()
                self.g.nodes[vertex]['visited_by'].remove(agent_id)
            else:
                self.visited_vertices.remove(vertex)
                self.unvisited_vertices.add(vertex)
                self.g.nodes[vertex]['visited_by'][0] = -1

    def getNeighborhood(self, vertex, radius):
        neighbors = set()
        for v in self.getNodes():
            if self.distance_matrix[vertex, v] < radius:
                neighbors.add(v)
        return neighbors

    def getCurrentRewardGain(self, vertex=-1):
        alpha = -np.log(0.01) / 2.0

        sum = 0.0
        if vertex != -1:
            for w in self.getNeighborhood(vertex, 2.0):
                if(w != vertex and w not in self.visited_vertices and w != 0 and
                        w != self.num_nodes-1):
                    sum += np.exp(-alpha * self.distance_matrix[vertex, w]) * self.getReward(w)

            sum += self.getReward(vertex)

        for v in self.visited_vertices:
            if v != 0 and v != self.num_nodes-1:
                for w in self.getNeighborhood(v, 2.0):
                    if(w != v and w not in self.visited_vertices and w != 0 and
                            w != self.num_nodes-1):
                        sum += np.exp(-alpha * self.distance_matrix[v, w]) * self.getReward(w)

                sum += self.getReward(v)

        return sum

    def computeGain(self, consider=[], desconsider=[], path2count=[]):

        alpha = -np.log(0.01) / 2

        utility = 0.0
        interest = set(self.visited_vertices)

        inter = set(desconsider).intersection(interest)
        interest = interest - inter
        interest.update(consider)

        if not path2count:
            path2count = interest

        for vertex in path2count:
            if vertex != 0 and vertex != self.num_nodes-1:
                for w in self.getNeighborhood(vertex, 2):
                    if w != vertex and w not in interest and w != 0 and w != self.num_nodes-1:
                        utility += np.exp(-alpha * self.distance_matrix[vertex, w]) * self.getReward(w)

                utility += self.getReward(vertex)

        return utility

    def computePathGain(self, p2dec, p2con):
        # exponential decay
        alpha = -np.log(0.01) / 2
        utility = 0.0

        path2deconsider = set(p2dec)
        path2consider = set(p2con)

        freenodes = set()
        for v in path2deconsider:
            if v not in path2consider:
                freenodes.add(v)

        for vertex in path2consider:
            if vertex != 0 and vertex != self.num_nodes-1:
                for w in self.getNeighborhood(vertex, 2):
                    if w != vertex and (w in self.unvisited_vertices or w in freenodes) and w != 0 and w != self.num_nodes-1:
                        utility += np.exp(-alpha * self.distance_matrix[vertex, w]) * self.getReward(w)

                utility += self.getReward(vertex)

        return utility

    def showSimpleEnv(self):
        pos = nx.get_node_attributes(self.g, 'position')
        nx.draw_networkx_nodes(self.g, pos,
                               node_color='black',
                               node_size=120,
                               alpha=0.8)

        nx.draw_networkx_labels(self.g, pos, font_size=8, font_color='w')

    def showEnv(self):
        fig, ax = plt.subplots()

        profits = set(nx.get_node_attributes(self.g, 'reward').values())
        mapping = dict(zip(sorted(profits), count()))
        nodes = self.g.nodes
        colors = [mapping[self.g.nodes[n]['reward']] for n in nodes]

        pos = nx.get_node_attributes(self.g, 'position')
        nc = nx.draw_networkx_nodes(self.g, pos,
                                    node_color=colors,
                                    node_size=120,
                                    alpha=0.8,
                                    cmap=plt.cm.jet)

        plt.colorbar(nc)

    def showPath(self, path, color, edge_alpha=0.5, edge_w=8, failed=False):
        if path:
            nx.draw_networkx_nodes(self.g, nx.get_node_attributes(self.g, 'position'),
                                   nodelist=path,
                                   node_color=color,
                                   node_size=120,
                                   alpha=0.5)
            # path edges
            edges = self.makePathEdges(path)
            nx.draw_networkx_edges(self.g, nx.get_node_attributes(self.g, 'position'),
                                   edgelist=edges[:-1],
                                   width=edge_w, alpha=edge_alpha, edge_color=color)

            if not failed:
                # last edge
                nx.draw_networkx_edges(self.g, nx.get_node_attributes(self.g, 'position'),
                                       edgelist=[edges[-1]],
                                       width=edge_w, alpha=edge_alpha, edge_color=color)
            else:
                nx.draw_networkx_nodes(self.g, nx.get_node_attributes(self.g, 'position'),
                                       nodelist=[path[-2]],
                                       node_color='black',
                                       node_shape="o",
                                       node_size=300,
                                       alpha=0.8)

                nx.draw_networkx_nodes(self.g, nx.get_node_attributes(self.g, 'position'),
                                       nodelist=[path[-2]],
                                       node_color=color,
                                       node_shape="P",
                                       node_size=120,
                                       alpha=0.8)
