import torch 
import pickle

from attention.selfAttention import SelfAttention
from attention.config import batch_size, output_size, hidden_size, vocab_size, embedding_length
# Load params
weights = torch.zeros((vocab_size, embedding_length))


def load_pytorch_model(modelPath):
    '''
    Function to load pytorch model.
    '''
    model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, 
                    embedding_length, weights)
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
