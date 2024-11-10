from typing import List, Tuple
from degree import density, main as degree_main
from math import comb


class Graph:
    def __init__(self, l: List[Tuple[int, int] | int]):
        assert isinstance(l, list)
        assert all(isinstance(x, tuple) or isinstance(x, int) for x in l)
        assert all(isinstance(x, int) or len(x) == 2 for x in l)
        assert all(isinstance(x, int) or isinstance(x[0], int) for x in l)
        assert all(isinstance(x, int) or isinstance(x[1], int) for x in l)
        assert all(isinstance(x, int) or x[0] != x[1] for x in l)
        V = set()
        for x in l:
            if isinstance(x, tuple):
                V.add(x[0])
                V.add(x[1])
            else:
                V.add(x)
        assert max({0}.union(V)) < len(V) or len(V) == 0
        assert min({0}.union(V)) >= 0
        adj = [[] for _ in range(len(V))]
        for x in l:
            if not isinstance(x, tuple): continue
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
    
    def triad_census(self):
        T = [0]*4
        m = sum(self.degrees())/2
        n = len(self.adj)
        adj = self.adj
        marked = set()
        vs = sorted(range(len(adj)), key=lambda x: len(adj[x]), reverse=True)
        index = {v: i for i, v in enumerate(vs)}
        Nplus = lambda v: [x for x in adj[v] if index[x] < index[v]]
        Nminus = lambda v: [x for x in adj[v] if index[x] > index[v]]
        for v in vs:
            T[2] += comb(len(adj[v]), 2)
            for u in Nminus(v):
                marked.add(u)
            for u in Nminus(v):
                for w in Nplus(u):
                    if w in marked:
                        T[3] += 1
            for u in Nminus(v):
                marked.remove(u)
        T[2] -= 3 * T[3]
        T[1] = m*(n-2) - 2*T[2] - 3*T[3]
        T[0] = comb(n, 3) - T[1] - T[2] - T[3]
        return T

    def degree(self, v: int):
        return len(self.adj[v])

    def degrees(self):
        return [len(x) for x in self.adj]
    
    def characteristic_path_length(self):
        adj = self.adj
        return self.wiener_index()/(len(adj)*(len(adj)-1))

    def wiener_index(self):
        adj = self.adj
        return sum(self.dist(u, v) for u in range(len(adj)) for v in range(len(adj)))


def main(data: List[Tuple[int, int]]): 
    g = Graph(data)
    fs = [
        g.degrees,
        g.cluster_coefficient,
        g.characteristic_path_length,
        g.wiener_index,
        g.triad_census,
    ]
    for f in fs:
        print(f.__name__)
        print(f'-->  {f()}')
        print()

    degree_main(g.degrees())
    

if __name__ == "__main__":
    # data = [
    #     (0, 1),
    #     (1, 2),
    #     (2, 3),
    #     (3, 4),
    #     (4, 5),
    #     (5, 0)
    # ]
    # data = [
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (0, 4),
    #     (0, 5)
    # ]
    data = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5)
    ]
    main(data)