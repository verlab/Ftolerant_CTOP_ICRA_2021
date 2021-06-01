"""Agent class."""
import sys
import pdb
from sortedcollections import ValueSortedDict


def removeVertex(env, visited, path):
    disp = [env.computeVertexUtility(v, visited) for v in path]
    if disp:
        min_ver = path[disp.index(min(disp))]
    else:
        min_ver = -1

    return min_ver


# structure used to optimize neighborhood analysis
class Pool(object):

    def __init__(self, env):
        self.visited = env.visited_vertices
        self.env = env
        self.nodes = [env.computeVertexDisponibility(x, self.visited) for x in env.getNodes('all')]
        self.edges = {}

    def updatePool(self, visited, path):
        self.visited = visited
        self.nodes = [self.env.computeVertexDisponibility(x, self.visited) for x in self.env.getNodes('all')]
        self.edges.clear()
        self.newEdges(path)

    def updatePoolSet(self, visited, path, upset):
        self.visited = visited
        vertices = set()

        for v in upset:
            vertices.update(self.env.getNeighborhood(v, 5))

        for v in vertices:
            self.nodes[v] = self.env.computeVertexDisponibility(v, self.visited)

        self.edges.clear()
        self.newEdges(path)

    def getNeighbor(self, edge, index):
        keys = edge.keys()
        return keys[index]

    def getBestNeighbors(self, edge):
        n = []
        try:
            for k in self.edges[edge].keys():
                if k not in self.visited:
                    n.append(k)
        except Exception:
            pass
        return n

    def newEdges(self, path):
        try:
            for e in self.env.makePathEdges(path):
                self.edges[e] = ValueSortedDict()
                for v in self.env.getNeighborhood(e[0], 5):
                    self.edges[e].update({v: -1*self.nodes[v]/(self.env.getVerticesDistance(e[0], v)+self.env.getVerticesDistance(v, e[1]))})
                for v in self.env.getNeighborhood(e[1], 5):
                    self.edges[e].update({v: -1*self.nodes[v]/(self.env.getVerticesDistance(e[0], v)+self.env.getVerticesDistance(v, e[1]))})
        except Exception:
            pdb.set_trace()

    def updateNodes(self, nodes, path):
        for v in nodes:
            self.nodes[v] = self.env.computeVertexDisponibility(v, self.visited)

        try:
            for e in self.edges.keys():
                for v in nodes:
                    if v in self.edges[e]:
                        self.edges[e].update({v: -1*self.nodes[v]/(self.env.getVerticesDistance(e[0], v)+self.env.getVerticesDistance(v, e[1]))})
        except Exception:
            pdb.set_trace()

    def insert(self, edge, node, path):
        self.visited.add(node)
        self.edges.pop(edge)
        new_path = [edge[0], node, edge[1]]
        self.newEdges(new_path)

        changes = self.env.getNeighborhood(node, 2)
        self.updateNodes(changes, path)

    def remove(self, edge1, edge2, path):
        node = edge1[1]
        if node != edge2[0]:
            pdb.set_trace()

        self.visited.remove(node)
        self.edges.pop((edge1[0], node))
        self.edges.pop((node, edge2[1]))

        new_path = [edge1[0], edge2[1]]
        self.newEdges(new_path)

        changes = self.env.getNeighborhood(node, 2)
        self.updateNodes(changes, path)


class Agent(object):
    """Agent class."""

    def __init__(self, environment, id, color, budget, nr, la, rem, pfails, path):
        """Instantiate the agent."""
        sys.stdout.write("Creating agent {:d}".format(id))
        self.id = id
        self.color = color
        self.budget = budget
        self.spended_budget = 0
        self.sensing_cost = 1.0  # we assume constant for all agents
        self.look_ahead = la
        self.neighborhood_range = nr
        self.percentage_remove = rem
        self.pfails = pfails
        self.atualizou = False

        self.path = path[:]
        self.planned_path = path[:]
        self.start_node = self.path[0]
        self.end_node = self.path[-1]
        self.moved_path = []

        self.path_index = -1
        self.position = -1
        self.last_position = -1
        self.state = 'planned'  # [planned  moving  failed  finished]

        if pfails == 100:
            self.stop_position = len(self.path) + 1  # will not fail
        else:
            self.stop_position = int((pfails*len(self.path))/100)

        self.rem = int(rem * la // 100)

        for v in self.path:
            environment.visit(v, self.id)

        self.path_cost = self.computePathCost(environment)
        sys.stdout.write(' ' + u'\u2713' + '\n')

    def computePathCost(self, environment, path=[], vset=[]):
        if not path:
            path = self.path
        if len(path) < 2:
            return 0.0

        lpath = path[:]
        for v in vset:
            lpath.remove(v)
        edge_set = environment.makePathEdges(lpath)

        cost = 0.0
        for edge in edge_set:
            cost += environment.getVerticesDistance(edge[0], edge[1])
            if edge[1] != self.end_node:
                cost += self.sensing_cost
            if path.count(edge[1]) > 1:
                cost += (path.count(edge[1])-1) * self.sensing_cost

        return cost

    def computePathReward(self, environment, path=[], vset=[]):
        if not path:
            path = self.path
        if len(path) < 2:
            return 0.0

        if not vset:
            vset = environment.visited_vertices

        rwd = 0.0
        for node in path:
            rwd += environment.computeVertexUtility(node, vset)

        return rwd

    def visit(self, env, vertex, insert=-1):
        env.visit(vertex, self.id)
        if insert == -1:
            self.path.append(vertex)
        else:
            self.path.insert(self.path.index(insert), vertex)

    def unvisit(self, env, vertex):
        env.unvisit(vertex, self.id)
        self.path.remove(vertex)
        self.path_cost = self.computePathCost(env)

    def setPool(self, environment):
        self.edge_pool = Pool(environment)

    def failed(self):
        if self.state == 'failed':
            return True
        return False

    def greedyExpand(self, environment, work_path, type='greedy', start_index=-1, end_index=-1):
        path_cost = self.computePathCost(environment, work_path)
        path = work_path
        if start_index == -1:
            start_index = self.path_index
        if end_index == -1:
            end_index = len(path)

        best = environment.computePathGain(path[start_index:end_index], path[start_index:end_index])
        best_candidate = -1
        best_edge = -1
        teste = ValueSortedDict()
        for edge in environment.makePathEdges(path[start_index:end_index]):
            neighbors = self.edge_pool.getBestNeighbors(edge)
            if neighbors:
                teste.update({edge: neighbors[0]})

        for edge in teste.keys():
            neighbors = teste[edge]
            old_distance = environment.getVerticesDistance(edge[0], edge[1])

            for candidate in [neighbors]:
                f_seg = environment.getVerticesDistance(edge[0], candidate)
                s_seg = environment.getVerticesDistance(candidate, edge[1])

                new_cost = path_cost - old_distance + f_seg + s_seg + self.sensing_cost

                if new_cost > self.budget:
                    continue

                newpath = path[start_index:end_index][:]
                newpath.insert(newpath.index(edge[1]), candidate)
                gain = environment.computePathGain(path, newpath)

                if gain > best:
                    if type == 'greedy':
                        best = gain

                    if type == 'first':
                        work_path.insert(work_path.index(edge[1]), candidate)
                        return

                    best_edge = edge
                    best_candidate = candidate

        if best_edge != -1:
            work_path.insert(work_path.index(best_edge[1]), best_candidate)
            self.edge_pool.insert(best_edge, best_candidate, work_path[start_index:end_index+1])

            xxx = self.computePathCost(environment, work_path)
            if xxx > self.budget:
                pdb.set_trace()

    def move(self, environment, agents):
        if self.state == 'finished' or self.state == 'failed':
            return

        # First move
        if self.path_index == -1:
            self.path_index = 0
            self.position = self.path[0]
            self.moved_path.append(self.position)
            self.state = 'moving'
            return

        self.last_position = self.position

        # Checks if it is the position where the agent will fail (calculated at agent instantiation)
        if self.path_index == self.stop_position:
            while self.path[self.path_index+1] != self.path[-1]:
                self.unvisit(environment, self.path[self.path_index+1])
            self.atualizou = True
            self.state = 'failed'

        self.path_index += 1
        self.position = self.path[self.path_index]
        self.spended_budget += environment.getVerticesDistance(self.last_position, self.position)
        if self.position != self.path[-1]:
            self.spended_budget += self.sensing_cost
        self.moved_path.append(self.position)

        if self.position == self.path[-1] and self.state != 'failed':
            self.state = 'finished'

    def adjustSegment(self, environment, teammates):
        if self.state == 'finished' or self.state == 'failed':
            return

        self.atualizou = False
        up = 0
        for ag in teammates:
            if ag.atualizou:
                up = 1
        if up == 0:
            return

        visited_others = set()
        for ag in teammates:
            if ag == self:
                continue
            visited_others.update(ag.path)

        visited = visited_others.union(set(self.path))
        work_path = self.path[:]

        last_lahead = self.path_index + self.look_ahead + 1
        size = len(self.path)
        if last_lahead > size:
            last_lahead = size

        upset = visited-self.edge_pool.visited
        upset.update(self.edge_pool.visited-visited)

        self.edge_pool.updatePoolSet(visited, self.path[self.path_index:last_lahead], upset)

        last_util = self.path_index + 1 + self.look_ahead  # TODO * self.teste
        if last_util > size:
            last_util = size
        last_util = size

        region = self.path[self.path_index:last_util]
        initial_objective = environment.computeGain(consider=visited,
                                                    desconsider=environment.visited_vertices,
                                                    path2count=region)

        last_rem = self.path_index + self.look_ahead
        if last_rem >= size:
            last_rem = -1

        icost = self.computePathCost(environment, work_path[self.path_index:last_rem])
        if icost == 0:
            icost += 1
        initial_value = initial_objective/icost

        for ky in range(self.rem):
            v = removeVertex(environment, self.edge_pool.visited, work_path[self.path_index+1:last_rem-ky])
            if v != -1:
                index = work_path.index(v)
                if index < len(work_path)-2:
                    v1 = work_path[index-1]
                    v2 = work_path[index+1]
                    work_path.remove(v)
                    self.edge_pool.remove((v1, v), (v, v2), work_path)

        # greedy_repair
        last_rep = self.path_index + self.look_ahead
        if last_rep >= len(work_path):
            last_rep = -1

        for ky in range(self.rem+2):
            self.greedyExpand(environment, work_path, start_index=self.path_index, end_index=last_rep)

        # compute gain
        visited = visited_others.union(set(work_path))
        last_util = self.path_index+1+self.look_ahead

        if last_util > len(work_path):
            last_util = len(work_path)
        last_util = len(work_path)
        region = work_path[self.path_index:last_util]
        final_objective = environment.computeGain(consider=visited,
                                                  desconsider=environment.visited_vertices,
                                                  path2count=region)

        last_rem = self.path_index + self.look_ahead
        if last_rem >= size:
            last_rem = -1

        fcost = self.computePathCost(environment, work_path[self.path_index:last_rem])
        if fcost == 0:
            fcost += 1
        final_value = final_objective/fcost

        if initial_value >= final_value:
            return

        self.atualizou = True
        region = self.path[self.path_index+1:last_lahead][:]
        for v in region:
            self.unvisit(environment, v)

        if self.path_index+1 == len(self.path):
            for v in work_path[self.path_index+1:]:
                self.visit(environment, v)
        else:
            point = self.path[self.path_index+1]

            for v in work_path[self.path_index+1:]:
                if v == point:
                    break
                self.visit(environment, v, insert=point)
