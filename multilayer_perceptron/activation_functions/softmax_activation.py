from typing import List

from multilayer_perceptron.activation_functions.base_exponent_activation import BaseExponentActivation


class SoftmaxActivation(BaseExponentActivation):

    def transform(self, numbers: List[float]) -> List[float]:
        exponent_list = [self.exponent_helper.calculate_exponent(number) for number in numbers]
        exponent_sum = sum(exponent_list)
        return [exponent/exponent_sum for exponent in exponent_list]
