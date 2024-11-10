from graph import Graph
from unittest import TestCase

class GraphSuite(TestCase):

    def test_graph(self):
        data = [
            (0, 1),
        ]
        g = Graph(data)
        self.assertEqual(g.adj, [[1], [0]])

    def test_graph(self):
        data = [
            (0, 1),
            (1, 2),
            (2, 0),
        ]
        g = Graph(data)
        self.assertEqual([set(x) for x in g.adj], [{1, 2}, {0, 2}, {0, 1}])

    def test_triad_census0(self):
        data = []
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [0, 0, 0, 0])

    def test_triad_census1(self):
        data = [0,1,2]
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [1, 0, 0, 0])
    
    def test_triad_census2(self):
        data = [
            (0, 1),
            2
        ]
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [0, 1, 0, 0])

    def test_triad_census3(self):
        data = [
            (0, 1),
            (1, 2),
        ]
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [0, 0, 1, 0])

    def test_triad_census4(self):
        data = [
            (0, 1),
            (1, 2),
            (2, 0),
        ]
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [0, 0, 0, 1])
    
    def test_triad_census5(self):
        data = [
            (0, 1),
            (1, 2),
            (2, 0),
            3
        ]
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [0, 3, 0, 1])
    
    def test_triad_census6(self):
        data = [
            (0, 1),
            (1, 2),
            (2, 0),
            3, 4
        ]
        g = Graph(data)
        result = g.triad_census()
        self.assertEqual(result, [3, 6, 0, 1])