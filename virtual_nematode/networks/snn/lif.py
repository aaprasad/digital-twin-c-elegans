import torch


class HeavisideStep(torch.autograd.Function):
    """ Heaviside step function """
    @staticmethod
    def forward(ctx, input, sigma):
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        # step function
        result = (input >= 1.) * 1.
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        # erfc gradient
        result = torch.exp(-(1. - input) ** 2 / (2. * sigma ** 2))
        return grad_output * result
