import nltk
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model import NeuralNet
from tqdm import tqdm
import pdb;


with open("intent.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (w, tag) in xy:
    bag = bag_of_words(w, all_words)
    X_train.append(bag)
    labels = tags.index(tag)
    y_train.append(labels)

X_train = np.array(X_train)
y_train = np.array(y_train)

# print(X_train.shape)  # 30x69
# print(y_train.shape)  # 30x1

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=8,
                          shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    for (words, labels) in train_loader:
        words = words.to(device=device)
        labels = labels.to(device=device)
        # pdb.set_trace()
        output = model(words)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

print(f"Final Loss: {loss.item():.4f}")

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

file_path = "model.pth"
torch.save(data, file_path)

print(f"Training completed!!!!")
