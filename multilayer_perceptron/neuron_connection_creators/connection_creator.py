from typing import List
from random import uniform

from multilayer_perceptron.models.neuron_connection import NeuronConnection
from multilayer_perceptron.network.dense_layer import DenseLayer


class NeuronConnectionCreator:

    def create_connections(self, input_layer: DenseLayer, output_layer: DenseLayer) -> List[NeuronConnection]:
        output_neuron_list = output_layer.create_neuron_types()

        return [
            NeuronConnection(input_neuron=input_neuron, output_neuron=output_neuron, weight=uniform(-0.25, 0.25))
            for input_neuron in input_layer.create_neuron_types() for output_neuron in output_neuron_list
        ]
