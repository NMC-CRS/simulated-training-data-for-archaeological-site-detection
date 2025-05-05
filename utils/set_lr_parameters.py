# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:28:51 2024

@author: Claudine Gravel-Miguel

"""

import torch.optim as optim

def setup_parameter_learning_rate(model):
    '''
    Set up the fine tuning of pretrained weights.

    Parameters
    ----------
    model : PyTorch model
        Actual model structure.

    Returns
    -------
    pretrained_optimizer : Pytorch optimizer
        Optimizer of the frozen parameters. Those do NOT get updated during training.
    new_optimizer : Pytorch optimizer
        Optimizer of the learning parameters, which gets updated during training.

    '''
                
    # Divide the frozen parameters and the ones that will learn into two categories
    new_params = [p for p in model.parameters() if p.requires_grad]
    pretrained_params = [p for p in model.parameters() if p.requires_grad == False]
    
    print(f"\n{len(new_params)} parameters will learn, whereas {len(pretrained_params)} remain frozen.\n")

    # Define the optimizer
    # Create separate optimizer instances for each set of parameters
    pretrained_optimizer = optim.AdamW(pretrained_params, lr=0)  # Set learning rate to 0 to freeze
    new_optimizer = optim.AdamW(new_params, lr=1e-3)  # Use a desired learning rate for the new layers
    
    # Return the frozen and learning parameters
    return pretrained_optimizer, new_optimizer

# THE END