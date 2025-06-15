import torch
from torch.distributions import Distribution, Normal, Uniform
import utils.pytorch_util as ptu


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample(n).detach()
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            self._clip_but_pass_gradient(1 - value * value) + self.epsilon
        )

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0., upper=1.):
        """Clipping function that allows for gradients to flow through.

        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        return x + clip
    def log_prob_sum(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        log_prob =self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )
        return log_prob.sum(dim=-1, keepdim=True)
    def log_prob_clip(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - torch.clamp(value * value, min=0, max=0.2)
        )

    def sample(self, return_pretanh_value=True):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=True):
        """
        Sampling in the reparameterization case.
        """

        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                ptu.zeros(self.normal_mean.size()),
                ptu.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample2a(self, return_pretanh_value=True):
        """
        Sampling in the reparameterization case.
        """
        Nrandom = Normal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                )
        r = Nrandom.sample()
        l = -r

        zr = (self.normal_mean + self.normal_std *r)

        zl = (self.normal_mean + self.normal_std *l)
        zr.requires_grad_()
        zl.requires_grad_()

        return torch.tanh(zr), zr, torch.tanh(zl), zl


    def rsamples(self, number=2, return_pretanh_value=True):
        """
        Sampling in the reparameterization case.
        """
        size = self.normal_mean.size()
        z = self.normal_mean + self.normal_std * Normal(ptu.zeros(size), ptu.ones(size)).sample([number])
        # z = z.reshape(-1, size[-1])

        # kk=torch.Size([2,2])
        # tmp = TanhNormal(torch.cuda.FloatTensor([[1, 10],[100,1000]]), torch.cuda.FloatTensor([[1, 1],[1,1]]))
        # tmp.normal_mean + tmp.normal_std * Normal(ptu.zeros(kk), ptu.ones(kk)).sample()

        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def samples(self, number=2, return_pretanh_value=True):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample([number]).detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)