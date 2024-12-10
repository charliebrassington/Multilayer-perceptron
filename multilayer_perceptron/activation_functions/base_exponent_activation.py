from multilayer_perceptron.activation_functions.base_activation import BaseActivation
from multilayer_perceptron.activation_functions._exponent_helper import _ExponentHelper

class BaseExponentActivation(BaseActivation):
    """Base class to provide an exponent helper to the classes which require exponent calcs"""
    def __init__(self, exponent_helper: _ExponentHelper =  _ExponentHelper()):
        self.exponent_helper = exponent_helper
