from typing import Dict, List
from collections import defaultdict

from multilayer_perceptron.models.neuron_connection import NeuronConnection
from multilayer_perceptron.neuron_types import AbstractNeuron


class UniqueConnectionCreator:

    def create_unique_connection_list(self, connection_list: List[NeuronConnection]) -> List[List[NeuronConnection]]:
        connection_dict = defaultdict(list)
        for connection in connection_list:
            connection_dict[connection.output_neuron].append(connection)

        return list(connection_dict.values())
