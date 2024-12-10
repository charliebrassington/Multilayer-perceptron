from abc import ABC, abstractmethod
from typing import List

from multilayer_perceptron.models.input_node import InputNode


class AbstractNeuron(ABC):
    """Abstract base class for all neuron types to add typing and remove duplicated logic"""

    def __init__(self, bias: float) -> None:
        self._bias = bias

    @abstractmethod
    def _process_input(self, weighted_sum: float) -> float:
        """
        Internal abstract method for the activation function logic to be held
        this is run after the weighted sum is calculated

        :param weighted_sum: float
        :return: float
        """
        raise NotImplementedError

    def process_input(self, input_node_list: List[InputNode]) -> float:
        """
        Process function for all neurons used in forward propagation.

        :param input_node_list: list of input node objects
        :return: float
        """
        weighted_sum = sum(input_node.input_feature * input_node.weight for input_node in input_node_list) + self._bias
        return self._process_input(weighted_sum=weighted_sum)
