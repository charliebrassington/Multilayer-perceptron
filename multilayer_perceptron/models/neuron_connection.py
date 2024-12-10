from dataclasses import dataclass

from multilayer_perceptron.neuron_types import AbstractNeuron


@dataclass
class NeuronConnection:
    input_neuron: AbstractNeuron
    output_neuron: AbstractNeuron
    weight: float
