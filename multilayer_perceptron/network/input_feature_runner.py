from typing import List

from multilayer_perceptron.network.dense_layer import DenseLayer
from multilayer_perceptron.models.neuron_connection import NeuronConnection
from multilayer_perceptron.models.input_node import InputNode

from multilayer_perceptron.neuron_connection_creators import NeuronConnectionListCreator, UniqueConnectionCreator
from multilayer_perceptron.network.input_feature_collector import InputFeatureCollector


class InputFeatureRunner:
    def __init__(
        self,
        unique_connection_creator: UniqueConnectionCreator = UniqueConnectionCreator(),
        input_feature_collector: InputFeatureCollector = InputFeatureCollector()
    ):
        self._unique_connection_creator = unique_connection_creator
        self._input_feature_collector = input_feature_collector

    def get_input_features(self, input_data: List[float], layer: int, connections: List[List[NeuronConnection]]) -> List[float]:
        unique_connection_lists = self._unique_connection_creator.create_unique_connection_list(
            connection_list=connections[layer]
        )

        return self._input_feature_collector.collect_input_feature_list(
            unique_connection_lists=unique_connection_lists, input_data=input_data
        )

