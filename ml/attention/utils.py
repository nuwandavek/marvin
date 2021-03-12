import torch 
import pickle

from attention.selfAttention import SelfAttention


def load_pytorch_model(model, modelPath):
    '''
    Function to load pytorch model.
    '''
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint)
    return model

def load_vocab_dict(dictPath):
    '''
    Function to load vocab dict.
    '''
    with open(dictPath, 'rb') as pickleDict:
    	vocab_dict = pickle.load(pickleDict)
    return vocab_dict
