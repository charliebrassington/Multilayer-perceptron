from typing import List

from multilayer_perceptron.neuron_types import AbstractNeuron, NEURON_TYPES


class DenseLayer:

    def __init__(self, neuron_count: int, activation_func: str) -> None:
        self._neuron_count = neuron_count
        self._activation_type_neuron = NEURON_TYPES[activation_func]

    def create_neuron_types(self) -> List[AbstractNeuron]:
        return [self._activation_type_neuron(bias=0.00) for _ in range(self._neuron_count)]
