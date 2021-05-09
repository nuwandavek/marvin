MODEL_PATHS = {
    "common":{
        "classifier_name" : "distilbert-base-uncased",
        "transfer_name" : "t5-small" 
    },
    'micro-formality' : {
        "classifier": "./models/distilbert_uncased_2/formality+emo/joint",
        "classifier_name": "distilbert-base-uncased",
        "classifier_nick": "distilbert_uncased_2",
        "classifier_task" : "formality+emo",
        "idx_to_classes" : {
            'formality': {'0': 'informal', '1': 'formal'},
            'emo': {'0': 'sad', '1': 'happy'}
        },
        "label_dims" : {'formality': 1, 'emo': 1},
        "transfer" : "./models/t5_transfer_formality_3",
        "transfer_name" : "t5-small",
        "transfer_nick" : "t5_transfer_formality_3",
    },
    'micro-joint' : {
        "classifier": "./models/distilbert_uncased_2/formality+emo/joint",
        "classifier_name": "distilbert-base-uncased",
        "classifier_nick": "distilbert_uncased_2",
        "classifier_task" : "formality+emo",
        "idx_to_classes" : {
            'formality': {'0': 'informal', '1': 'formal'},
            'emo': {'0': 'sad', '1': 'happy'}
        },
        "label_dims" : {'formality': 1, 'emo': 1},
        "transfer" : "./models/t5_transfer_formality_joint",
        "transfer_name" : "t5-small",
        "transfer_nick" : "t5_transfer_formality_joint",
    },
    'macro-shakespeare' : {
        "classifier": "./models/distilbert_uncased_2/shakespeare/joint",
        "classifier_name": "distilbert-base-uncased",
        "classifier_nick": "distilbert_uncased_2",
        "classifier_task" : "shakespeare",
        "idx_to_classes" : {
            'shakespeare': {'0': 'noshakespeare', '1': 'shakespeare'}
        },
        "label_dims" : {'shakespeare': 1},
        "transfer" : "./models/t5_transfer_shakespeare",
        "transfer_name" : "t5-small",
        "transfer_nick" : "t5_transfer_shakespeare",
    },
    'macro-binary' : {
        "transfer_name" : "t5-small",
        "transfer_shake" : "./models/t5_transfer_shakespeare_binary",
        "transfer_nick_shake" : "t5_transfer_shakespeare_binary",
        "transfer_abs" : "./models/t5_transfer_abstract_binary",
        "transfer_nick_abs" : "t5_transfer_abstract_binary",
        "transfer_wiki" : "./models/t5_transfer_wiki_binary",
        "transfer_nick_wiki" : "t5_transfer_wiki_binary",
    }
}
