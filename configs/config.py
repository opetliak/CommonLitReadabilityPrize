# data config
TRAIN_DATA = 'data/train.csv'
TEST_SIZE = 0.3
# TEST_DATA = 'data/test.csv' # for kaggle


# train config
BASE_MODEL = "camembert-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 2
NUM_LABELS = 1
SAVE_METRIC = "mse"
SAVE_PATH = "checkpoints/camembert-fine-tuned-regression"
METRIC_PATH = 'checkpoints/eval_metrics.json'

DEVICE = "cuda"