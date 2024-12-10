from multilayer_perceptron.neuron_types.abstract_neuron import AbstractNeuron
from multilayer_perceptron.activation_functions import SigmoidActivation


class SigmoidNeuron(AbstractNeuron):

    def __init__(self, bias: float, sigmoid_activation_function: SigmoidActivation = SigmoidActivation()):
        super().__init__(bias)
        self._sigmoid_activation_function = sigmoid_activation_function

    def _process_input(self, weighted_sum: float) -> float:
        return self._sigmoid_activation_function.transform(weighted_sum)
