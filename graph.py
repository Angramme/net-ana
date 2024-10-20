from typing import List, Tuple
from degree import density, main as degree_main


class Graph:
    def __init__(self, l: List[Tuple[int, int]]):
        assert isinstance(l, list)
        assert all(isinstance(x, tuple) for x in l)
        assert all(len(x) == 2 for x in l)
        assert all(isinstance(x[0], int) for x in l)
        assert all(isinstance(x[1], int) for x in l)
        assert all(x[0] != x[1] for x in l)
        V = set()
        for x in l:
            V.add(x[0])
            V.add(x[1])
        assert max(x) < len(V) and min(x) >= 0
        adj = [[] for _ in range(len(V))]
        for x in l:
            adj[x[0]].append(x[1])
            adj[x[1]].append(x[0])
        self.adj = adj

    def cluster_coefficient(self, v: int|None=None):
        adj = self.adj
        if v is None:
            return sum(self.cluster_coefficient(v) for v in range(len(adj)))/len(adj)
        assert v < len(adj) and v >= 0
        neighbors = set(adj[v])
        degrees = [sum(1 for y in adj[x] if y in neighbors) for x in adj[v]]
        return density(degrees)
    
    def dist(self, u: int, v: int):
        adj = self.adj
        assert u < len(adj) and u >= 0
        assert v < len(adj) and v >= 0
        q = [u]
        dist = [-1]*len(adj)
        dist[u] = 0
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist[v]
    
    def degrees(self):
        return [len(x) for x in self.adj]
    
    # Wiener’s index
    def characteristic_path_length(self):
        adj = self.adj
        return sum(self.dist(u, v) for u in range(len(adj)) for v in range(len(adj)))/(len(adj)*(len(adj)-1))
    

def main(data: List[Tuple[int, int]]): 
    g = Graph(data)
    fs = [
        g.cluster_coefficient,
        g.characteristic_path_length,
        g.degrees
    ]
    for f in fs:
        print(f.__name__)
        print(f'-->  {f()}')
        print()

    degree_main(g.degrees())
    