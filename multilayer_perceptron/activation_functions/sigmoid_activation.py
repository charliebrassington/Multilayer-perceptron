from multilayer_perceptron.activation_functions.base_exponent_activation import BaseExponentActivation


class SigmoidActivation(BaseExponentActivation):

    def transform(self, number: float) -> float:
        return 1 / (1 + self.exponent_helper.calculate_exponent(-number))
