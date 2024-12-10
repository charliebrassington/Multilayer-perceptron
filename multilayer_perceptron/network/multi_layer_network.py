from typing import List
from collections import defaultdict

from multilayer_perceptron.network.dense_layer import DenseLayer

from multilayer_perceptron.neuron_connection_creators import NeuronConnectionListCreator
from multilayer_perceptron.network.input_feature_runner import InputFeatureRunner


class MultiLayerNetwork:

    def __init__(
        self,
        layers: List[DenseLayer],
        connection_creator: NeuronConnectionListCreator = NeuronConnectionListCreator(),
        input_feature_runner: InputFeatureRunner = InputFeatureRunner()
    ):
        self._connections = connection_creator.create_list_connections(layers)
        self._input_feature_runner = input_feature_runner

    def start_forward_propagation(self, input_data: List[float], layer: int = 0):
        if layer == len(self._connections):
            return input_data

        input_features = self._input_feature_runner.get_input_features(input_data, layer, self._connections)
        return self.start_forward_propagation(input_data=input_features, layer=layer+1)
