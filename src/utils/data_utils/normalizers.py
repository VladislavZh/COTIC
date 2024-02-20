import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import fsolve


class Normalizer(ABC):
    """
    Abstract base class for normalizers.
    """

    @abstractmethod
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data.
        """
        pass

    @abstractmethod
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes the data.
        """
        pass

    @classmethod
    @abstractmethod
    def from_data(cls, data: torch.Tensor) -> "Normalizer":
        """
        Creates a normalizer from data.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, float]:
        """
        Returns a dictionary containing the state of the normalizer.

        Returns:
        dict[str, float]: A dictionary with keys and values representing the stateful elements of the normalizer.
        """
        pass

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state_dict: dict[str, float]) -> "Normalizer":
        """
        Loads the state from the provided dictionary into a new normalizer instance.

        Parameters:
        state_dict (Dict[str, float]): A dictionary containing the state of the normalizer.

        Returns:
        Normalizer: A new instance of a Normalizer subclass with its state loaded from the dictionary.
        """
        pass


class ExponentialNormalizerP99(Normalizer):
    """
    A normalizer class that scales data from an exponential distribution with
    an unknown rate parameter to an Exponential(1) distribution and vice versa.
    """

    def __init__(self, lambda_value: float) -> None:
        """
        Initializes the normalizer with the given lambda value.

        Parameters:
        lambda_value (float): The rate parameter of the original exponential distribution.
        """
        self.lambda_value = lambda_value

    @classmethod
    def from_data(cls, data: torch.Tensor) -> "Normalizer":
        """
        Class method to create an ExponentialNormalizer from data.

        Parameters:
        data (torch.Tensor): A tensor of data points from an exponential distribution.

        Returns:
        An instance of ExponentialNormalizer initialized with the computed lambda value.
        """
        # Calculate the 99th percentile
        x_99 = torch.quantile(data, 0.99).item()

        # Estimate E_trunc using the mean of the truncated data
        truncated_data = data[data <= x_99]
        E_trunc = truncated_data.mean().item()

        # Initial guess for lambda could be 1/x_99 for simplicity
        lambda_initial_guess = 1 / x_99

        # Use the provided method to solve for lambda
        lambda_value = cls.solve_for_lambda(E_trunc, x_99, lambda_initial_guess)

        return cls(lambda_value)

    @staticmethod
    def solve_for_lambda(E_trunc: float, x_99: float, lambda_initial_guess: float) -> float:
        """
        Solves for the rate parameter lambda of an exponential distribution.

        Parameters:
        E_trunc (float): The expectation of X below the 99th percentile.
        x_99 (float): The 99th percentile of the distribution.
        lambda_initial_guess (float): An initial guess for the lambda value.

        Returns:
        The solved rate parameter lambda.
        """

        def equation(lambda_value):
            return E_trunc - (1 / lambda_value - x_99 / (np.exp(lambda_value * x_99) - 1))

        # Solve for lambda using fsolve
        lambda_solution = fsolve(equation, lambda_initial_guess)

        return lambda_solution[0]

    def normalize(self, data_point: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a data point to an Exponential(1) distribution.

        Parameters:
        data_point (torch.Tensor): The data point to normalize.

        Returns:
        The normalized data point.
        """
        normalized_data_point = data_point * self.lambda_value
        return normalized_data_point

    def denormalize(self, normalized_data_point: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes a data point back to the original Exponential(lambda) distribution.

        Parameters:
        normalized_data_point (torch.Tensor): The data point to denormalize.

        Returns:
        The denormalized data point.
        """
        denormalized_data_point = normalized_data_point / self.lambda_value
        return denormalized_data_point

    def state_dict(self) -> dict[str, float]:
        """
        Returns a dictionary containing the state of the normalizer.

        Returns:
        Dict[str, float]: A dictionary with a key 'lambda_value' and its corresponding value.
        """
        return {'lambda_value': self.lambda_value}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, float]) -> "Normalizer":
        """
        Loads the state from the provided dictionary into a new ExponentialNormalizerP99 instance.

        Parameters:
        state_dict (dict[str, float]): A dictionary containing the state of the normalizer.

        Returns:
        Normalizer: A new instance of ExponentialNormalizerP99 with its state loaded from the dictionary.
        """
        return cls(state_dict['lambda_value'])
