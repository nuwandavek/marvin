def salience(sent, model, tokenizer, device, task, label_num):
    '''
    Runs the model on an input sentence and gives the salience for each token that 
    is a representation of how much a given token is contributing to the score of a 
    given class for a given classification task.
    
    Arguments: 
    
      sent       : str. 
                   The input sentence.
      model      : the PyTorch sequence classifier model.  
                   Assumed to be a JointSeqClassifier or 
                   DistilBertForSequenceClassification model
      tokenizer  : Transformers tokenizer.
                   The tokenizer used to encode the input string. 
                   Assumed to be compatible with model.
      device     : torch.device. 
                   The device to be used (cuda or cpu)
      task   : str 
                   string representing the task
      label_num  : int.
                   Index for the class label for the task corresponding to task_num.
                   For example if task 0 is a politeness classificaition 
                   task and class 1 is "polite", then setting task_num = 0 and 
                   label_num = 1 would give us a representation of how much each 
                   token is contributing to labelling the sentence as "polite".
    Returns:
    
      saliency : Tensor of floats with length equal to length of tokenized sent.
                 The gradient of the particular output we are interested in (the score for 
                 class 'label_num' for task 'task_num') wrt the input tokens.
    '''
    # Set model to evaluation mode
    model.eval()
    # Tokenize input
    input_tensor = tokenizer.encode(sent, return_tensors="pt").to(device)
    
    
    # Run the model and get the output
    output = model(input_tensor, output_attentions=True, output_hidden_states=True)
    
    print(output)
    
    loss_dict, logits_dict, selected_labels_dict, hidden_states = output
    
    print(logits_dict)
    
    # Get the score for the desired task,class combination
    class_score = logits_dict[task][0][label_num]
    # backprop
    class_score.backward()
    # get the gradients
    grads = [h_state.grad for h_state in hidden_states]
    # Take the max over the heads and the mean over the embedding dim. 
    # TODO, consider different ways to do this
    saliency = torch.stack(grads).abs().max(axis=0)[0].squeeze().mean(axis=1)
    return saliency
