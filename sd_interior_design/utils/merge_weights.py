import torch


def weighted_sum(theta0, theta1, alpha):
    return ((1 - alpha) * theta0) + (alpha * theta1)


# Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
def sigmoid(theta0, theta1, alpha):
    alpha = alpha * alpha * (3 - (2 * alpha))
    return theta0 + ((theta1 - theta0) * alpha)


# Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
def inv_sigmoid(theta0, theta1, alpha):
    import math

    alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
    return theta0 + ((theta1 - theta0) * alpha)


@torch.no_grad()
def merge(
    m1: torch.nn.Module, m2: torch.nn.Module, alpha: float = 0.5, merge_fn=weighted_sum
):
    """Merge the weights of two models.

    m1 is modified in-place, and then returned for convenience.

    Args:
        m1 (torch.nn.Module): First model to merge.
        m2 (torch.nn.Module): Second model to merge.
        alpha (float, optional): Parameter that controls the weight of m1 vs m2.
                                 alpha=0.0 returns m1, alpha=1.0 returns m2.
                                 Defaults to 0.5.
    """
    m1_weights = m1.state_dict()
    m2_weights = m2.state_dict()

    if m1_weights.keys() != m2_weights.keys():
        raise Exception("Tried to merge models with incompatible weights.")

    for key in m1_weights.keys():
        m1_weights[key] = merge_fn(m1_weights[key], m2_weights[key], alpha)

    m1.load_state_dict(m1_weights)
    return m1
