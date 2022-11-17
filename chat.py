import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intent.json', 'r') as f:
    intents = json.load(f)

file_path = "model.pth"
data = torch.load(file_path)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

# for key, value in model_state.items():
#     print(key)
#     print(value.size())

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Alice"


def get_response(msg):
    msg = tokenize(msg)
    X = bag_of_words(msg, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return f"{random.choice(intent['responses'])}"
    else:
        return f"I dont understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
