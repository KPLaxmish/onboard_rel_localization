import numpy as np

# Basic
# def weighted_mse_loss(input, target, weight):
#     return (weight * (input - target) ** 2).mean()

def weighted_mse_loss(input, target, weight):
    return (10 * weight * (input - target) ** 2).sum()/input.shape[0]

def mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum()/input.shape[0]