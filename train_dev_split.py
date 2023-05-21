import json
import os
import sklearn
from sklearn.model_selection import train_test_split

data_root = "data/"

# with open("data/legal_corpus.json", "r",encoding="utf-8") as db:
#     text = json.load(db)

# print(text[0])

with open(os.path.join(data_root, "train_question_answer.json"), 'r', encoding="utf-8") as f:
    train_data = json.load(f)
    train_data = train_data['items']

train, dev = train_test_split(train_data, test_size=0.2)

with open("data/train_split_data.json", "w", encoding="utf-8") as fw:
    json.dump(train, fw, indent=4, ensure_ascii=False)

with open("data/dev_split_data.json", "w", encoding="utf-8") as fw:
    json.dump(dev, fw, indent=4, ensure_ascii=False)