# This is a sample Python script.
from collections import defaultdict

import numpy
import pandas as pandas
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.optim as optim

from torch.utils.data import DataLoader

import Dota_Data
import Dota_Predictor
import FullyConnected
import Handwritten_Numbers
import pandas as pd
import Model_Manager
torch.set_printoptions(threshold=10_000)
if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU name:", torch.cuda.get_device_name(0))
    print("PyTorch version:", torch.__version__)
else:
    print("GPU is not available.")
def do_number_model():

    cross_entropy_loss = nn.CrossEntropyLoss()
    #model = FullyConnected.CustomModel(input_size1=input_size, output_size=output_size, criterion=cross_entropy_loss, activation=nn.ReLU())



    #print("loss: ", loss)

    #Handwritten_Numbers.load_data_one_hot()

    data = Handwritten_Numbers.load_data_one_hot()
    for epoch in range(40):
        model.train_dataset(data.get("train"), 1)
        loss: float = model.evaluate(data.get("test"))

        print("loss: ", loss)

def do_dota_model():
    data = Dota_Data.load_data_as_one_hot()
    duplicate_count = count_duplicates(data.get("train"), data.get("test"))
    print("dub count: ", duplicate_count)
    model = None
    while(True):
        model = Dota_Predictor.Main_Model(criterion=nn.CrossEntropyLoss(), activation=nn.ReLU())
        #model = Dota_Predictor.Perceptron_Model(criterion=nn.CrossEntropyLoss(), activation=nn.ReLU())
        loss = 0.0
        for epoch in range(1):
            loss = Dota_Predictor.train_dataset(model, data.get("train"), 10)
            #loss = Dota_Predictor.evaluate(model, data.get("test"))
            print("loss: ", loss)
        if(loss < 0.65):
            break
    for epoch in range(1):
        Dota_Predictor.train_dataset(model, data.get("train"), 50)
        loss = Dota_Predictor.evaluate(model, data.get("test"))
        print("loss: ", loss)

def count_duplicates(train_dataloader, test_dataloader):
    data_dict = defaultdict(lambda: {"train": 0, "test": 0})

    for features, labels in train_dataloader:
        for i in range(features.size(0)):
            key = (tuple(features[i].flatten().tolist()), tuple(labels[i].tolist()))
            data_dict[key]["train"] += 1

    for features, labels in test_dataloader:
        for i in range(features.size(0)):
            key = (tuple(features[i].flatten().tolist()), tuple(labels[i].tolist()))
            data_dict[key]["test"] += 1

    duplicate_count = 0
    for key, value in data_dict.items():
        if value["train"] > 0 and value["test"] > 0:
            duplicate_count += min(value["train"], value["test"])

    return duplicate_count

def analyze_data():
    train_file = "train.txt"
    test_file = "test.txt"
    data = Dota_Data.load_data_as_one_hot()
    train_data = data.get("train").dataset
    test_data = data.get("test").dataset

    train_features, train_labels = zip(*[(torch.argmax(x, dim=1).tolist(), y) for x, y in train_data])
    test_features, test_labels = zip(*[(torch.argmax(x, dim=1).tolist(), y) for x, y in test_data])

    train_df = pd.DataFrame(list(zip(train_features, train_labels)), columns=['feature', 'label'])
    test_df = pd.DataFrame(list(zip(test_features, test_labels)), columns=['feature', 'label'])

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    duplicate_count = count_duplicates(data.get("train"), data.get("test"))
    print("dub count: ", duplicate_count)

if __name__ == "__main__":
    do_dota_model()