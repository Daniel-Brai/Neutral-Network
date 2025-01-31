"""
A library to load the MNIST image data.
"""

from typing import Any, Tuple, List
import pickle
import gzip
import numpy as np
from numpy import ndarray

from .utils import vectorized_result


def load_data() -> (
    Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]
):
    """
    Return the MNIST data as a tuple containing the training data, the validation data, and the test data.
    """
    f = gzip.open("./data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper() -> Tuple[Any, Any, Any]:
    """
    Return a tuple containing ``(training_data, validation_data, test_data)``
    """
    tr_d, va_d, te_d = load_data()
    training_inputs: List[ndarray] = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results: List[ndarray] = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs: List[ndarray] = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs: List[ndarray] = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)