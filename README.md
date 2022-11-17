Simple custom rasa chatbot implementation with Pytorch<br/>

## Installation
### Clone this repo
```console
git clone https://github.com/tienv1per/custom_rasa_chatbot.git
```

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv


### Activate

```console
. venv/bin/activate
```

### Install PyTorch and dependencies
You also need `nltk`:
 ```console
pip install nltk
 ```
 
 If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run 
```console
python3 train.py
```
This will dump `model.pth` file. And then run
```console
python3 chat.py
```

## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
