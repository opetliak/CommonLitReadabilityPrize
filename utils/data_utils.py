from configs import config
from datasets import DatasetDict
from functools import partial

def preprocess_function(examples, tokenizer):
    label = examples["target"] 
    examples = tokenizer(examples["excerpt"], truncation=True,
                         padding="max_length", max_length=config.MAX_LENGTH)
    examples["label"] = label
    return examples


def prepare_train_test_data(raw_ds, tokenizer):

    train_test_valid = raw_ds.train_test_split(config.TEST_SIZE) #30% for test + valid
    test_valid = train_test_valid['test'].train_test_split(test_size=0.5)# 15% + 15%

    train_test_valid_dataset = DatasetDict({
        'train': train_test_valid['train'],
        'test': test_valid['train'],
        'valid': test_valid['test']})


    for split in train_test_valid_dataset:
        train_test_valid_dataset[split] = train_test_valid_dataset[split].map(partial(preprocess_function, tokenizer=tokenizer),
                                                                              remove_columns=["id",
                                                                                              "url_legal",
                                                                                              "license",
                                                                                              'url_legal',
                                                                                              'excerpt',
                                                                                              'standard_error', 
                                                                                              'target'])
    return train_test_valid_dataset
    
        
    