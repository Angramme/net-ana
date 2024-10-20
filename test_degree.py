from degree import *
import unittest

class Suite(unittest.TestCase):

    def test_prepare_degree(self):
        data = [1, 2, 3, 4, 5]
        with self.assertRaises(AssertionError):
            prepare_degrees(data)
    
    def test_prepare_degree_with_negative(self):
        data = [1, 2, 3, 4, -5]
        with self.assertRaises(AssertionError):
            prepare_degrees(data)

    def test_h_index(self):
        data = [2, 3, 2, 2, 7, 5, 4, 3, 4, 2]
        result = h_index(data)
        self.assertTrue(result == 4)
    
    def test_h_index2(self):
        data = [3, 5, 2, 4, 3, 7, 2, 2]
        result = h_index(data)
        self.assertTrue(result == 4)

    def test_splittance(self):
        data = [2, 3, 2, 2, 7, 5, 4, 3, 4, 2]
        result = splittance(data)
        self.assertTrue(result == 3.0)

    def test_splittance2(self):
        data = [3, 4, 2, 5, 2, 4, 4, 3, 3, 2]
        result = splittance(data)
        self.assertTrue(result == 5.0)
    
    def test_splittance3(self):
        data = [3, 5, 2, 4, 3, 7, 2, 2]
        result = splittance(data)
        self.assertTrue(result == 1.0)

    def test_kth_erdos_gallai(self):
        data = [3, 5, 2, 4, 3, 7, 2, 2]
        degrees = prepare_degrees(data)
        for k in range(2, len(degrees)):
            left, right = kth_erdos_gallai(degrees, k)
            self.assertTrue(left < right)
        left, right = kth_erdos_gallai(degrees, 1)
        self.assertEqual(left, right)

    def test_edge_count(self):
        data = [3,4,2,2,5,4,3,4,3,2]
        result = count_edges(data)
        self.assertTrue(result == 16)