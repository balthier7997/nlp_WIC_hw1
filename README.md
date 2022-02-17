# NLP-2021: First Homework
This is the first homework of the NLP 2021 course at Sapienza University of Rome ([course info](http://naviglinlp.blogspot.com/)).
- Instructor: Roberto Navigli ([webpage](http://wwwusers.di.uniroma1.it/~navigli/))
- Teaching assistant: Cesare Campagnano
- Teaching assistant: Pere-Lluís Huguet Cabot

## Requirements

* Ubuntu distribution 19.10 or the current LTS are perfectly fine.
* [conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community

## Setup Environment

1. Install Docker
2. Setup a client
3. Download `glove.6B.50d.txt` embeddings from [this page](https://nlp.stanford.edu/projects/glove/) and put it inside `model/embeddings`

For those interested, *test.sh* essentially setups a server exposing your model through a REST Api and then queries this server, evaluating your model.

### Install Docker

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Logout and re-login (**important**). 
```
newgrp docker
sudo service docker restart
```

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```
conda create -n nlp2021-hw1 python=3.7
conda activate nlp2021-hw1
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```
conda activate nlp2021-hw1
bash test.sh data/dev.jsonl
```

Actually, you can replace *data/dev.jsonl* to point to a different file, as far as the target file has the same format.

If you hadn't changed *hw1/stud/model.py* yet when you run test.sh, the scores you just saw describe how a random baseline.
