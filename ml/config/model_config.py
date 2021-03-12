
model_type = "bert"
task_name = 'politeness'
#"self_attention"

def get_configs():
	if model_type == 'self_attention':
		configs = {'model_type' : 'self_attention',
		'vocab_size' : 100000 + 2,
		'embedding_length' : 100,
		'hidden_size' : 100,
		'output_size' : 2,
		'batch_size' : 32,
		'max_length' : 40,
		'vocab_dict_path' :  "../ml/models/attention/vocab_dict.pickle",
		'model_path' : "../ml/models/attention/attn_test.pth"}
		
	elif model_type == 'bert':
		configs = {'model_type' : 'bert',
		 		   'batch_size' : 32,
			       'model_path' : "../ml/models/BERT"}
			       
	if task_name == 'politeness':
		class_labels_dict = {0: "impolite", 1 : "polite"}
	return configs, class_labels_dict
