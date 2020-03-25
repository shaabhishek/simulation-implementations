import numpy as np

class BaseDistribution:
    def __init__(self):
        pass

    def _sample(self):
        """Method to return the random variables with basic computation"""
        raise NotImplementedError

class Distribution(BaseDistribution):
    def __init__(self):
        pass

    def _sample(self):
        return None

    def sample(self, *shape):
        num_samples = np.prod(shape, dtype=int)
        samples = [self._sample() for _ in range(num_samples)]
        samples = np.reshape(samples, shape)
        return samples