# CommonLit Readability Prize

## Project overview
1. Trained (fine-tuned) model for a [kaggle task](https://www.kaggle.com/c/commonlitreadabilityprize/overview)
2. API for testing the results


## How to use
This will automatically train the model and run server using FastAPI
```
git clone 
sh build.sh
```
Than you can run this in a separate console
```
curl -X POST "http://127.0.0.1:8889/regression" -H  "Content-Type: application/json"  -d '{"text":"bla"}'
```
Expected output
```{"score":[0.05607207864522934]} ```

Typical metrics will be (print during the training, and saved to ```checkpoints/eval_metrics.json```)
| Eval_mse  | Eval_mae  | Eval_rmse | 
| ------------- | ------------- | ------------- |
| 0.85 | 0.74 | 0.92 |