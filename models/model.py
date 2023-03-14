from transformers import TrainingArguments
from configs import config
from transformers import AutoModelForSequenceClassification, AutoTokenizer


training_args = TrainingArguments(
        output_dir=config.SAVE_PATH,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model=config.SAVE_METRIC,
        load_best_model_at_end=False,
        weight_decay=0.01)

def build_model():
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL,
                                                               num_labels=config.NUM_LABELS)
    
    return model, tokenizer