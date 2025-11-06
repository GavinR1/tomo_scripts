"""
Basic test cases for data utilities.
"""

import pytest
import numpy as np
import os
from src.utils.data_utils import normalize_data

def test_normalize_data():
    # Test with regular data
    test_data = np.array([1, 2, 3, 4, 5])
    normalized = normalize_data(test_data)
    assert normalized.min() == 0
    assert normalized.max() == 1
    
    # Test with constant data
    constant_data = np.ones((3, 3))
    normalized = normalize_data(constant_data)
    assert np.all(normalized == 0)
    
    # Test with empty data
    with pytest.raises(ValueError):
        normalize_data(np.array([]))