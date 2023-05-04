import decimal

import torch
from typing import Any
import torch.nn as nn
from torch import optim
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

import FullyConnected


class Main_Model(nn.Module):
    def __init__(self, criterion: nn.Module,
                 activation: nn.Module = nn.ReLU):
        super(Main_Model, self).__init__()
        self.criterion = criterion
        self.activation = activation

        self.hero_model = FullyConnected.Fully_Connected_Model(input= 140, output= 16, inner_layer= 32, num_layers=3, activation=activation)

        self.counter = FullyConnected.Fully_Connected_Model(input= 32, output= 8, inner_layer= 16, num_layers=3, activation=activation)

        self.team_model = FullyConnected.Fully_Connected_Model(input=16 * 5 + 4, output= 16, inner_layer= 64, num_layers=3, activation=activation)

        self.final_model = FullyConnected.Fully_Connected_Model(input= 32, output= 2, inner_layer= 16, num_layers=3, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #10 tensors
        tensors_tuple = torch.split(x, 1, dim=1)
        #10 one-hot tensors
        tensors_tuple = [torch.squeeze(tensor, 1) for tensor in tensors_tuple]  # remove the second dimension to get (64, 140)

        #10 tensors
        output_tensors = [self.hero_model(tensor) for tensor in tensors_tuple]

        #combinations
        combinations = [(i, j) for i in range(5) for j in range(5, 10)] + [(i, j) for i in range(5, 10) for j in range(5)]
        #concatenate combinations
        concatenated_tensors = [torch.cat((output_tensors[i], output_tensors[j]), dim=1) for i, j in combinations]
        counters = [self.counter(tensor) for tensor in concatenated_tensors]
        #print("counter: ", counters[0].shape)

        # Apply max pooling to each group of combinations
        max_pool = MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        #ally heroes compared to enemy heroes
        group1 = torch.stack(counters[:25], dim=1)
        #enemy heroes compared to ally heroes
        group2 = torch.stack(counters[25:], dim=1)

        #group comparisons
        pooled_group1, _ = torch.max(max_pool(group1), dim=1)
        pooled_group2, _ = torch.max(max_pool(group2), dim=1)

        #the first and last 5 tensors are grouped
        first_half = torch.cat(output_tensors[:5], dim=1)
        first_half = torch.cat([first_half, pooled_group1], dim=1)
        #concatenate grouped heroes and comparisons
        second_half = torch.cat(output_tensors[5:], dim=1)
        second_half = torch.cat([second_half, pooled_group2], dim=1)

        #team vectors
        output1 = self.team_model(first_half)
        output2 = self.team_model(second_half)

        # Concatenate the output tensors
        teams_cat = torch.cat((output1, output2), dim=1)
        final_output = self.final_model(teams_cat)

        softmax = nn.Softmax(dim = 1)

        return softmax(final_output)

class Perceptron_Model(nn.Module):
    def __init__(self, criterion: nn.Module,
                 activation: nn.Module = nn.ReLU()):
        super(Perceptron_Model, self).__init__()

        hero_vector = 32

        self.criterion = criterion
        self.activation = activation




        self.team_model = FullyConnected.Fully_Connected_Model(input=140 * 10, output= 128, inner_layer= 200, num_layers=3, activation=activation)

        self.final_model = FullyConnected.Fully_Connected_Model(input= 128, output= 2, inner_layer= 32, num_layers=3, activation=activation)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensors_tuple = torch.split(x, 1, dim=1)
        # 10 tensors (64, 140)
        tensors_tuple = [torch.squeeze(tensor, 1) for tensor in tensors_tuple]
        #print("tuple shape: ", tensors_tuple[0].shape)
        x = torch.cat(tensors_tuple[:], dim=1)
        #print("x shape: ", x.shape)
        x = self.team_model(x)
        x = self.final_model(x)
        x = nn.Softmax()(x)
        return x



def train_dataset(model, dataloader: DataLoader, epochs: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer: torch.optim.Optimizer = optim.Adam(model.parameters(), lr=0.0005)
    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        count = 0
        #print("epoch: ", epoch)
        #training in batches
        for x, y in dataloader:
            #print("step")
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            #over all batch data, get the result array
            y_pred = model.forward(x)

            #y = torch.argmax(y, dim=1)

            #calculate the loss
            loss = model.criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            loss_sum += loss
            count += 1

        losses.append(loss_sum.item()/count)
    decimal.getcontext().prec = 3
    print(losses)
    return losses[-1]

def evaluate(model, dataloader: DataLoader) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    #input whether each instance was correct where y is the index
    correct = [0,0]
    incorrect = [0,0]

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model.forward(x)
            loss = model.criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            # Get the predicted classes
            true_class = torch.argmax(y, dim=1)
            predicted_classes = torch.argmax(y_pred, dim=1)

            #print(y.shape)
            #print(y_pred.shape)
            # Update correct and incorrect arrays
            print("y: ", y )
            print("y pred", y_pred)
            for i in range(len(true_class)):
                if predicted_classes[i] == true_class[i]:
                    correct[true_class[i]] += 1
                else:
                    incorrect[true_class[i]] += 1
    print("correct", correct)
    print("incorrect", incorrect)
    model.train()
    return total_loss / total_samples

