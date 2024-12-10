from multilayer_perceptron.activation_functions.base_activation import BaseActivation


class ReluActivation(BaseActivation):

    def transform(self, number: int) -> int:
        return number if number > 0 else 0
