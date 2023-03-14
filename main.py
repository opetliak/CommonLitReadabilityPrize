from fastapi import FastAPI
from pydantic import BaseModel, constr, conlist
from typing import List
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from configs import config
import os
import glob


app = FastAPI()

list_of_ckpt = glob.glob(f'{config.SAVE_PATH}/*')
latest_ckpt = max(list_of_ckpt, key=os.path.getctime)

tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(latest_ckpt,
                                                           local_files_only=True).to(config.DEVICE)

def predict(input_texts):
    encoded = tokenizer(input_texts,
                        truncation=True,
                        padding="max_length",
                        max_length=config.MAX_LENGTH,
                        return_tensors="pt").to(config.DEVICE)

    pred = model(**encoded).logits.reshape(-1).tolist()
    return {'score': pred}


class UserRequestIn(BaseModel):
    text: constr(min_length=1)

class ScoredLabelsOut(BaseModel):
    score: List[float]

@app.post("/regression", response_model=ScoredLabelsOut)
def read_classification(user_request_in: UserRequestIn):
    return predict(user_request_in.text)