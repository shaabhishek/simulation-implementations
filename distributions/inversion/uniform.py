import numpy as np

from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean

class Uniform(Distribution):
    def __init__(self, a:float, b:float, ugen:RngStream):
        self.a = a
        self.b = b
        self.width = b-a
        self.mean = (a+b)/2
        self.var = self.width**2 / 12
        self.ugen = ugen
    
    @property
    def mean(self):
        return self.mean

    @property
    def var(self):
        return self.var

    def _sample(self):
        u = self.ugen.RandU01()
        return self.a + self.width*u
    
    def sample(self, *shape):
        num_samples = np.prod(shape, dtype=int)
        samples = [self._sample() for _ in range(num_samples)]
        samples = np.reshape(samples, shape)
        return samples

class Uniform01(Uniform):
    def __init__(self, ugen:RngStream):
        super().__init__(0,1, ugen)

    def _sample(self):
        return self.ugen.RandU01()

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    num_samples = args.num_samples

    ugen = RngStream()
    # unif_dist = Uniform01(ugen=ugen)
    unif_dist = Uniform(100, 500, ugen=ugen)

    samples = unif_dist.sample(num_samples, 100)

    DensityPlotter(dist=unif_dist).make_plots(samples.flatten())
    test_mean(unif_dist, samples)