"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  
"""

import json
import random
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import sigmoid, sigmoid_prime, vectorized_result


class QuadraticCost:
    @staticmethod
    def fn(a: NDArray, y: NDArray) -> Any:
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z: NDArray, a: NDArray, y: NDArray) -> NDArray:
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a: NDArray, y: NDArray) -> float:
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z: NDArray, a: NDArray, y: NDArray) -> NDArray:
        return a - y


class Network:
    """
    The Neural Network
    """

    def __init__(self, sizes: List[int], cost=CrossEntropyCost) -> None:
        """
        Initialize the network with the sizes (the number of neurons in the
        respective layers of the network).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self) -> None:
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self) -> None:
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a: NDArray) -> NDArray:
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        training_data: List[Tuple[NDArray, NDArray]],
        epochs: int,
        mini_batch_size: int,
        eta: float,
        lmbda: float = 0.0,
        evaluation_data: Optional[List[Tuple[NDArray, NDArray]]] = None,
        monitor_evaluation_cost: bool = False,
        monitor_evaluation_accuracy: bool = False,
        monitor_training_cost: bool = False,
        monitor_training_accuracy: bool = False,
    ) -> Tuple[List[float], List[int], List[float], List[int]]:
        """Train the neural network using mini-batch stochastic gradient descent."""

        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_evaluation_cost and evaluation_data:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy} / {n_data}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(
        self,
        mini_batch: List[Tuple[NDArray, NDArray]],
        eta: float,
        lmbda: float,
        n: int,
    ) -> None:
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x: NDArray, y: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
        """
        Return the gradient for the cost function.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: List[Tuple[NDArray, NDArray]]) -> int:
        """
        Return the number of test inputs for which the neural network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations: NDArray, y: NDArray) -> NDArray:
        """
        Returns the cost derivative
        """
        return output_activations - y

    def accuracy(
        self, data: List[Tuple[NDArray, NDArray]], convert: bool = False
    ) -> int:
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(
        self, data: List[Tuple[NDArray, NDArray]], lmbda: float, convert: bool = False
    ) -> Any:
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(int(y.item()))
            cost += self.cost.fn(a, y) / len(data)
        cost += (
            0.5
            * (lmbda / len(data))
            * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        )
        return cost

    def save(self, filename: str) -> None:
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__),
        }

        f = open(filename, "w")
        json.dump(data, f)
        f.close()

        print("\n\nNetwork data saved successfully\n")


def load_saved_network_from_file(filename: str) -> Network:
    with open(filename, "r") as f:
        data = json.load(f)
    cost = globals()[data["cost"]]
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
