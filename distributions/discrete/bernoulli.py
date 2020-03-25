from rngStream import RngStream

from distributions.distribution import Distribution
from distributions.make_plots import DensityPlotter
from distributions.utils import get_parser, test_mean

class Bernoulli(Distribution):
    def __init__(self, p:float, ugen:RngStream):
        self.p = p
        self.ugen = ugen

    @property
    def mean(self):
        return self.p

    @property
    def var(self):
        return self.p * (1-self.p)

    def _sample(self):
        u = self.ugen.RandU01()
        return int(u < self.p)


def setup_distribution():
    ugen = RngStream()
    dist = Bernoulli(p=.7, ugen=ugen)
    return dist

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    num_samples = args.num_samples

    dist = setup_distribution()

    samples = dist.sample(num_samples)
    DensityPlotter(dist=dist).make_plots(samples.flatten())
    test_mean(dist, samples)