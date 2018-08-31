class Node(object):
    def __init__(self, value):
        self.value = value
        self.edges = []
        self.visited = False
        self.dist = 100000*10

class Edge(object):
    def __init__(self, value, node_from, node_to):
        self.value = value
        self.node_from = node_from
        self.node_to = node_to

# You only need to change code with docs strings that have TODO.
# Specifically: Graph.dfs_helper and Graph.bfs
# New methods have been added to associate node numbers with names
# Specifically: Graph.set_node_names
# and the methods ending in "_names" which will print names instead
# of node numbers

class Graph(object):
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []
        self.node_names = []
        self._node_map = {}

    def set_node_names(self, names):
        """The Nth name in names should correspond to node number N.
        Node numbers are 0 based (starting at 0).
        """
        self.node_names = list(names)

    def insert_node(self, new_node_val):
        "Insert a new node with value new_node_val"
        new_node = Node(new_node_val)
        self.nodes.append(new_node)
        self._node_map[new_node_val] = new_node
        return new_node

    def insert_edge(self, new_edge_val, node_from_val, node_to_val):
        "Insert a new edge, creating new nodes if necessary"
        nodes = {node_from_val: None, node_to_val: None}
        for node in self.nodes:
            if node.value in nodes:
                nodes[node.value] = node
                if all(nodes.values()):
                    break
        for node_val in nodes:
            nodes[node_val] = nodes[node_val] or self.insert_node(node_val)
        node_from = nodes[node_from_val]
        node_to = nodes[node_to_val]
        new_edge = Edge(new_edge_val, node_from, node_to)
        node_from.edges.append(new_edge)
        node_to.edges.append(new_edge)
        self.edges.append(new_edge)

    def get_edge_list(self):
        return [(e.value, e.node_from.value, e.node_to.value)
                for e in self.edges]

    def get_edge_list_names(self):
        return [(edge.value,
                self.node_names[edge.node_from.value],
                self.node_names[edge.node_to.value])
                for edge in self.edges]

    def get_adjacency_list(self):
        max_index = self.find_max_index()
        adjacency_list = [[] for _ in range(max_index)]
        for edg in self.edges:
            from_value, to_value = edg.node_from.value, edg.node_to.value
            adjacency_list[from_value].append((to_value, edg.value))
        return [a or None for a in adjacency_list] # replace []'s with None

    def get_adjacency_list_names(self):
        adjacency_list = self.get_adjacency_list()
        def convert_to_names(pair, graph=self):
            node_number, value = pair
            return (graph.node_names[node_number], value)
        def map_conversion(adjacency_list_for_node):
            if adjacency_list_for_node is None:
                return None
            return map(convert_to_names, adjacency_list_for_node)
        return [map_conversion(adjacency_list_for_node)
                for adjacency_list_for_node in adjacency_list]

    def get_adjacency_matrix(self):
        """Return a matrix, or 2D list.
        Row numbers represent from nodes,
        column numbers represent to nodes.
        Store the edge values in each spot,
        and a 0 if no edge exists."""
        max_index = self.find_max_index()
        adjacency_matrix = [[0] * (max_index) for _ in range(max_index)]
        for edg in self.edges:
            from_index, to_index = edg.node_from.value, edg.node_to.value
            adjacency_matrix[from_index][to_index] = edg.value
        return adjacency_matrix

    def find_max_index(self):
        """Return the highest found node number
        Or the length of the node names if set with set_node_names()."""
        if len(self.node_names) > 0:
            return len(self.node_names)
        max_index = -1
        if len(self.nodes):
            for node in self.nodes:
                if node.value > max_index:
                    max_index = node.value
        return max_index

    def find_node(self, node_number):
        "Return the node with value node_number or None"
        return self._node_map.get(node_number)

    def _clear_visited(self):
        for node in self.nodes:
            node.visited = False

    def _clear_dist(self):
        for node in self.nodes:
            node.dist = 100000*10

    def get_all_dist(self):
        return [node.dist for node in self.nodes]

    def get_dist(self, nodes):
        return [node.dist for node in nodes]

    def __dijkstras(self, start_node_num):
        self._clear_dist()
        start_node = ret_val = self.find_node(start_node_num)
        start_node.dist = 0
        temp_nodes = self.nodes.copy()
        while len(temp_nodes):
            for edge in start_node.edges:
                if start_node is not edge.node_to:
                    current_dist = edge.node_to.dist
                    new_dist = start_node.dist + edge.value
                    if new_dist < current_dist:
                        edge.node_to.dist = new_dist
            temp_nodes.remove(start_node)
            distances = self.get_dist(temp_nodes)
            if distances:
                min_value_index = distances.index(min(distances))
                start_node = temp_nodes[min_value_index]
        return ret_val


    def distance_map(self, start_node_num,end_node_num):
        node = self.__dijkstras(start_node_num)
        name = self.node_names[node.value]
        for i in self.nodes:
            print(name, ' to ', self.node_names[i.value], ' ----> ', i.dist)




graph = Graph()


graph.set_node_names(('U',  # 0
                      'D',  # 1
                      'A',  # 2
                      'C',  # 3
                      'I',  # 4
                      'T',  # 5
                      'Y')) # 6

graph.set_node_names(('U',  # 0
                      'D',  # 1
                      'A',  # 2
                      'C',  # 3
                      'I',  # 4
                      'T',  # 5
                      'Y')) # 6


graph.insert_edge(4, 0, 1)  # U <-> D
graph.insert_edge(4, 1, 0)  # D <-> U
graph.insert_edge(3, 0, 2)  # U <-> A
graph.insert_edge(3, 2, 0)  # A <-> U
graph.insert_edge(7, 0, 3)  # U <-> C
graph.insert_edge(7, 3, 0)  # C <-> U
graph.insert_edge(3, 1, 3)  # D <-> C
graph.insert_edge(3, 3, 1)  # C <-> D
graph.insert_edge(6, 2, 4)  # A <-> I
graph.insert_edge(6, 4, 2)  # I <-> A
graph.insert_edge(4, 3, 4)  # C <-> I
graph.insert_edge(4, 4, 3)  # I <-> C
graph.insert_edge(2, 3, 5)  # C <-> T
graph.insert_edge(2, 5, 3)  # T <-> C
graph.insert_edge(4, 4, 6)  # I <-> Y
graph.insert_edge(4, 6, 4)  # Y <-> I
graph.insert_edge(5, 5, 6)  # T <-> Y
graph.insert_edge(5, 6, 5)  # Y <-> T




graph.distance_map(5)
