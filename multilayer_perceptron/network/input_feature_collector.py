from typing import List

from multilayer_perceptron.network.input_node_creator import InputNodeCreator
from multilayer_perceptron.models.neuron_connection import NeuronConnection

class InputFeatureCollector:

    def __init__(self, input_node_creator: InputNodeCreator = InputNodeCreator()):
        self._input_node_creator = input_node_creator

    def collect_input_feature_list(self, unique_connection_lists: List[List[NeuronConnection]], input_data: List[float]) -> List[float]:
        return [
            connection_list[0].output_neuron.process_input(
                input_node_list=self._input_node_creator.collect_input_node_list(connection_list, input_data)
            )
            for connection_list in unique_connection_lists
        ]
