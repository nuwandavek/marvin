import numpy as np
import torch 

def sent_pred(sent, model, tokenizer, device, batch_size):
    '''
    Runs the model on an input sentence.
    
    Arguments: 
    
      sent : str. The input sentence.
      model : the pytorch model to be used.
     
    Returns:
      pred : np array. The prediction, wich is a normalized array with a value for \
             each class, representing the predicted probability for that class
      attns : the attention matrix
    '''
    input_tensor = tokenizer.encode(sent, return_tensors="pt").to(device)

    output = model(input_tensor, output_attentions=True)
    pred = output.logits.argmax(axis=1)
    
    softmax = torch.nn.Softmax(dim=1)
    scores = softmax(output.logits.detach())
    
    attns = output.attentions
    
    return pred.detach().cpu().numpy(), scores, attns
    
    
def process_outputs(pred, scores, attns):
  	# Sum over attention vectors for each head and handle dimensions and move to cpu
	viz_attns = np.array([attn.sum(axis=1).cpu().detach().squeeze().numpy() for attn in attns])
	# Sum over heads
	viz_attns = viz_attns.sum(axis=0)
	# Drop cls and sep tokens
	viz_attns = viz_attns[0, 1:-1].tolist()
	
	
	scores = scores.cpu().detach().squeeze().numpy()
	
	pred = pred[0]
	
	return pred, scores, viz_attns
