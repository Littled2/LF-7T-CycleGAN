"""

Implementation of the content adversarial loss, as described by Yang et al. in: https://doi.org/10.1109/TMI.2025.3597401

"""

import torch



def content_adversarial_loss(G, D, x, y):
    """
    Computes the content adversarial loss

    """

    return torch.log(1 - D(G(x))).mean() + torch.log(D(y)).mean()



