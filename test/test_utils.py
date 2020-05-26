import pytest

from src.utils import merge_dictionaries_with_disagreements, matching_items

class Test_matching_items():

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


class Test_merge_dictionaries_with_disagreements():

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