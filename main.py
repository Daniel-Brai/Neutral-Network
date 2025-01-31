import json
import random

import numpy as np

from src.mnist_loader import load_data_wrapper
from src.network import CrossEntropyCost, Network, load_saved_network_from_file

SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def main() -> None:
    # train_network()
    evaluate_network()


def train_network() -> None:
    random.seed(12345678)
    np.random.seed(12345678)

    training_data, validation_data, _ = load_data_wrapper()
    # Initialize the network with 784 input layer, 140 hidden layer & 10 output layer
    net = Network([784, 100, 40, 10], cost=CrossEntropyCost)

    # Initialize the network with 784 input layer, 30 hidden layer & 10 output layer
    # net = Network([784, 30, 10], cost=CrossEntropyCost)
    accuracies = []

    for size in SIZES:
        print(f"\n\nTraining network with dataset size {size}")
        net.large_weight_initializer()
        num_epochs = 1500000 / size
        net.SGD(
            training_data[:size],
            int(num_epochs),
            10,
            0.5,
            lmbda=size * 0.0001,
            evaluation_data=None,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True,
        )
        accuracy = net.accuracy(validation_data) / 100.0
        print(f"Network Accuracy is {accuracy} percent")
        accuracies.append(accuracy)

    f = open("network_accuracy_784_100_40_10.json", "w")
    json.dump(accuracies, f)
    f.close()

    net.save("network_data_784_100_40_10.json")


def evaluate_network() -> None:
    _, _, test_data = load_data_wrapper()

    net = load_saved_network_from_file("network_data_784_100_40_10.json")
    test_result = net.evaluate(test_data)
    print(
        f"Network evaluated the {test_result} out of {len(test_data)} test data accurately"
    )


if __name__ == "__main__":
    main()
