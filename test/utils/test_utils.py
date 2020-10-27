import numpy as np
from bitarray import bitarray

from src.utils import merge_dictionaries_with_disagreements, matching_items, subset, normalize, Orientation, \
    get_points_to_plot


class Test_matching_items:

    def test_two_match(self):
        d1 = {1: True, 2: True,  4: False}
        d2 = {1: True, 2: True, 3: False}
        expected = [1, 2]

        result = matching_items(d1, d2)
        assert expected == result

        result = matching_items(d2, d1)
        assert expected == result

    def test_one_match(self):
        d1 = {1: True, 2: True,  4: False}
        d2 = {1: False, 2: True, 3: False}
        expected = [2]

        result = matching_items(d1, d2)
        assert expected == result

        result = matching_items(d2, d1)
        assert expected == result

    def test_zero_match(self):
        d1 = {1: True, 2: False,  4: False}
        d2 = {1: False, 2: True, 3: False}
        expected = []

        result = matching_items(d1, d2)
        assert expected == result

        result = matching_items(d2, d1)
        assert expected == result


class Test_merge_dictionaries_with_disagreements:

    def test_no_drop(self):
        d1 = {1: True, 4: False}
        d2 = {2: True, 3: False}
        expected = {**d1, **d2}

        result = merge_dictionaries_with_disagreements(d1, d2)
        assert expected == result

        result = merge_dictionaries_with_disagreements(d2, d1)
        assert expected == result

    def test_one_drop(self):

        d1 = {1: True, 2: False}
        d2 = {2: True, 3: False}
        expected = {1: True, 3: False}

        result = merge_dictionaries_with_disagreements(d1, d2)
        assert expected == result

        result = merge_dictionaries_with_disagreements(d2, d1)
        assert expected == result

    def test_all_drop(self):

        d1 = {1: True, 2: False}
        d2 = {1: False, 2: True}
        expected = {}
        
        result = merge_dictionaries_with_disagreements(d1, d2)
        assert expected == result

        result = merge_dictionaries_with_disagreements(d2, d1)
        assert expected == result

    def test_all_but_one_drop(self):

        d1 = {1: True, 2: False}
        d2 = {1: False, 2: True, 3: True}
        expected = {3: True}
        
        result = merge_dictionaries_with_disagreements(d1, d2)
        assert expected == result

        result = merge_dictionaries_with_disagreements(d2, d1)
        assert expected == result

    def test_one_empty(self):

        d1 = {}
        d2 = {1: False, 2: True, 3: True}
        expected = d2
        
        result = merge_dictionaries_with_disagreements(d1, d2)
        assert expected == result

        result = merge_dictionaries_with_disagreements(d2, d1)
        assert expected == result

    def test_both_empty(self):

        d1 = {}
        d2 = {}
        expected = {}
        
        result = merge_dictionaries_with_disagreements(d1, d2)
        assert expected == result

        result = merge_dictionaries_with_disagreements(d2, d1)
        assert expected == result


class Test_rest:

    def test_subset(self):
        a = bitarray('1000111')
        b = bitarray('1100010')
        c = bitarray('1100111')

        assert subset(a, b) is False
        assert subset(a, c) is True
        assert subset(c, a) is False
        assert subset(b, c) is True
        assert subset(c, b) is False

    def test_normalize(self):
        a = np.random.randint(0, 100, 10)

        b = normalize(a)

        assert min(b) == 0
        assert max(b) == 1
        assert len(a) == len(b)
        assert all(np.argsort(a) == np.argsort(b))

        c = np.array([5, 5, 5, 5])

        assert all(normalize(c) == np.array([1, 1, 1, 1]))


class Test_orientation:

    def test_init(self):
        o1 = Orientation(True)
        o2 = Orientation(False)
        o3 = Orientation('both')

        assert o1.direction == 'left'
        assert o2.direction == 'right'
        assert o3.direction == 'both'
        assert (o1 == o2) is False
        assert (o3 == o2) is False
        assert (o1 == o3) is False

    def test_change(self):
        o1 = Orientation(True)
        o3 = Orientation('both')

        assert o3.direction == 'both'

        o3.direction = 'left'

        assert (o1 == o3)

    def test_string(self):
        o1 = Orientation(True)
        o2 = Orientation(False)
        o3 = Orientation('both')

        assert o1.__str__() == 'True'
        assert o2.__str__() == 'False'
        assert o3.__str__() == 'both'


class Test_points_to_plot:

    def test_2d_with_cs(self):
        xs = np.random.rand(100, 2)
        cs = np.random.rand(2, 2)

        xs_out, cs_out = get_points_to_plot(xs, cs)

        for xs_i, xs_out_i in zip(xs, xs_out):
            assert all(xs_i == xs_out_i)

        for cs_i, cs_out_i in zip(cs, cs_out):
            assert all(cs_i == cs_out_i)

    def test_2d_no_cs(self):
        xs = np.random.rand(100, 2)
        cs = None

        xs_out, cs_out = get_points_to_plot(xs, cs)

        for xs_i, xs_out_i in zip(xs, xs_out):
            assert all(xs_i == xs_out_i)
        assert cs == cs_out

    def test_more_d_with_cs(self):
        xs = np.random.rand(100, 5)
        cs = np.random.rand(3, 5)

        xs_out, cs_out = get_points_to_plot(xs, cs)

        assert xs_out.shape == (100, 2)
        assert cs_out.shape == (3, 2)

    def test_more_d_no_cs(self):
        xs = np.random.rand(100, 5)
        cs = None

        xs_out, cs_out = get_points_to_plot(xs, cs)

        assert xs_out.shape == (100, 2)
        assert cs == cs_out


