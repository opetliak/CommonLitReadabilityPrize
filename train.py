from configs import config
from datasets import Dataset
from utils import model_utils, data_utils
from models.model import build_model, training_args
import json

def train():
    
    model, tokenizer = build_model()
    
    raw_ds = Dataset.from_csv(config.TRAIN_DATA)
    train_data = data_utils.prepare_train_test_data(raw_ds, tokenizer)

    
    
    trainer = model_utils.RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data["train"],
        eval_dataset=train_data["test"],
        compute_metrics=model_utils.compute_metrics_for_regression)
    
    trainer.train()
    trainer.eval_dataset=train_data["valid"]
    testing_metrics = trainer.evaluate()
    with open(config.METRIC_PATH, "w") as outfile:
        json.dump(testing_metrics, outfile)


if __name__ == "__main__":
    train()



