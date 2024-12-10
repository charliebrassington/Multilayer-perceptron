from typing import List

from multilayer_perceptron.models.input_node import InputNode
from multilayer_perceptron.models.neuron_connection import NeuronConnection


class InputNodeCreator:

    def collect_input_node_list(self, connection_list: List[NeuronConnection], data_list: List[float]) -> List[InputNode]:
        return [
            InputNode(input_feature=data, weight=connection.weight)
            for connection, data in zip(connection_list, data_list)
        ]
