from typing import Type, Dict

from multilayer_perceptron.neuron_types.abstract_neuron import AbstractNeuron
from multilayer_perceptron.neuron_types.sigmoid_neuron import SigmoidNeuron


NEURON_TYPES: Dict[str, Type[AbstractNeuron]] = {
    "sigmoid": SigmoidNeuron
}
