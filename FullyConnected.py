# Define the custom model
import torch
from typing import Any
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader



class Fully_Connected_Model(nn.Module):
    def __init__(self, input: int, output: int, num_layers: int,
                 activation: nn.Module = nn.ReLU(), inner_layer: int = 200):
        super(Fully_Connected_Model, self).__init__()

        self.activation = activation

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if(i == 0):
                self.layers.append(nn.Linear(input, inner_layer))
                continue
            if(i == num_layers - 1):
                self.layers.append(nn.Linear(inner_layer, output))
                continue
            self.layers.append(nn.Linear(inner_layer, inner_layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

class CustomModel(nn.Module):
    def __init__(self, input_size1: int, output_size: int, criterion: nn.Module,
                 activation: nn.Module, inner_layer_size: int = 200):

        super(CustomModel, self).__init__()

        self.criterion = criterion
        self.activation = activation

        # Sub-network 1
        self.fc1: nn.Linear = nn.Linear(input_size1, inner_layer_size)

        self.inner_model = Fully_Connected_Model(input= inner_layer_size, output= inner_layer_size, inner_layer= inner_layer_size, num_layers=10, activation=activation)

        # Sub-network 2
        self.fc2: nn.Linear = nn.Linear(inner_layer_size, inner_layer_size)

        # Combined network
        self.fc3: nn.Linear = nn.Linear(inner_layer_size, output_size)
        #self.fc4: nn.Softmax = nn.Softmax(1)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #x = self.forward(x)
        x = (self.inner_model.forward(x))
        x = (self.fc3(x))
        #x = self.fc4(x)
        x = x.reshape(x.size(0), -1)
        return x

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        optimizer: torch.optim.Optimizer = optim.Adam(self.parameters(), lr=0.0005)
        optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_dataset(self, dataloader: DataLoader, epochs: int) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer: torch.optim.Optimizer = optim.Adam(self.parameters(), lr=0.001)
        print("batch size: ", dataloader.batch_size)
        print("len: ", dataloader.__len__())
        #retrained over the same data
        for epoch in range(epochs):
            print("epoch: ", epoch)
            #training in batches
            for x, y in dataloader:
                #print("step")

                x, y = x.to(device), y.to(device)


                optimizer.zero_grad()
                #over all batch data, get the result array
                y_pred = self.forward(x)
                #print("y_shape: ", y.shape)
                #print("pred_shape: ", y_pred.shape)
                #y = torch.argmax(y, dim=1)
                #print("y after argmax, ", y.shape, "ex: ", y[0])

                #calculate the loss
                loss = self.criterion(y_pred, y)
                #print("loss: ",loss)

                #calculate the gradient
                # changing the parameters based on the gradient
                loss.backward()
                optimizer.step()
                break

    def evaluate(self, dataloader: DataLoader) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        #input whether each instance was correct where y is the index
        correct = [0,0,0,0,0,0,0,0,0,0]
        incorrect = [0,0,0,0,0,0,0,0,0,0]

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = self.forward(x)
                loss = self.criterion(y_pred, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

                # Get the predicted classes
                true_class = torch.argmax(y, dim=1)
                predicted_classes = torch.argmax(y_pred, dim=1)

                #print(y.shape)
                #print(y_pred.shape)
                # Update correct and incorrect arrays
                for i in range(len(true_class)):
                    if predicted_classes[i] == true_class[i]:
                        correct[true_class[i]] += 1
                    else:
                        incorrect[true_class[i]] += 1

        print("correct", correct)
        print("incorrect", incorrect)
        self.train()
        return total_loss / total_samples

