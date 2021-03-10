import torch

def sent_embedd(sent, model, vocab_dict, tokenizer, max_length, device, unk_embedd=True):
    '''
    Helper function to extract the embedding for an input sentence.
    '''
    idxs = [0 for i in range(max_length)]
    i = 0
    for word in tokenizer(sent):
        if i < max_length:
            if word in vocab_dict:
                idxs[i] = vocab_dict[word]
            else:
                # If using <unk> embedding, append 
                # the final index where that embedding is stored
                if unk_embedd:
                    idxs[i] = len(vocab_dict)
            i += 1
    return torch.LongTensor([idxs]).to(device)
            
def sent_pred(sent, model, vocab_dict, tokenizer, max_length, device, batch_size):
    '''
    Runs the model on an input sentence.
    
    Arguments: 
    
      sent : str. The input sentence.
      model : the pytorch model to be used.
      vocab_dict : dict. A dictionary with words as keys and their indices as values
     
    Returns:
      pred : np array. The prediction, wich is a normalized array with a value for \
             each class, representing the predicted probability for that class
      attns : the attention matrix
    '''
    input_tensor = sent_embedd(sent, model, vocab_dict, tokenizer, max_length, device)
    input_tensor = torch.cat(batch_size * [input_tensor])
    pred, attns = model(input_tensor, return_attn=True)
    return pred.detach().cpu().numpy(), attns
