""" LIF neuron model """

import torch


class HeavisideStep(torch.autograd.Function):
    """ Heaviside step function
    forward: O(t) = U(v(t) - V_th)
    backward: dU(v)/dv ~= dP(v)/dv = exp(-(v - V_th) ** 2 / (2 * sigma ** 2)) / (sqrt(2 * pi) * sigma)
        where P(v) = erfc(-(v - V_th) / (sqrt(2) * sigma)) / 2
            ~= U(v) = U(v - V_th)
    reference: Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network
    """
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
