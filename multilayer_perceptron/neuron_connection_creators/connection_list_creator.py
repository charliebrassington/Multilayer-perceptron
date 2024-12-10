from typing import List

from multilayer_perceptron.models.neuron_connection import NeuronConnection
from multilayer_perceptron.network.dense_layer import DenseLayer
from multilayer_perceptron.neuron_connection_creators.connection_creator import NeuronConnectionCreator


class NeuronConnectionListCreator:

    def __init__(self, neuron_connection_creator: NeuronConnectionCreator = NeuronConnectionCreator()):
        self._neuron_connection_creator = neuron_connection_creator

    def create_list_connections(self, layer_list: List[DenseLayer]) -> List[List[NeuronConnection]]:
        return [
            self._neuron_connection_creator.create_connections(input_layer=layer, output_layer=layer_list[index + 1])
            for index, layer in enumerate(layer_list)
            if index+1 < len(layer_list)
        ]
