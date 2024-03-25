import uvicorn
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from serialization import *
from models.vgg import *
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from typing import Dict
from utils.loading import DatasetSplit
from utils.training import distill
import os
import torch
import json

class Request(BaseModel):
    student_model: str # base 64 encoding of the model

app = FastAPI()

@app.get("/rpc")
def fl_round(request: Request) -> Dict[str, str]:
    
    # load environment variables (initialized in main.py)
    device = os.environ["device"]
    s_lr = float(os.environ["s_lr"])
    s_momentum = float(os.environ["s_momentum"])
    s_weight_decay = float(os.environ["s_weight_decay"])
    lr = float(os.environ["lr"])
    momentum = float(os.environ["momentum"])
    weight_decay = float(os.environ["weight_decay"])
    local_ep = int(os.environ["local_ep"])
    bs = int(os.environ["bs"])

    # load student model
    student_model.load_state_dict(deserialize_model(request.student_model, device))

    # training
    student_model.to(device)
    student_model.train()
    model.train()
    student_optimizer = torch.optim.SGD(
        student_model.parameters(), 
        s_lr, 
        s_momentum, 
        s_weight_decay,
    )
    model_optimizer = torch.optim.SGD(
        student_model.parameters(), 
        lr, 
        momentum, 
        weight_decay,
    )
    trainloader = DataLoader(DatasetSplit(trainset, dataset_indices), batch_size=bs, shuffle=True)

    for _ in range(local_ep):
        for _, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)

            distill(model_optimizer, student_model, model, X, y)
            distill(student_optimizer, model, student_model, X, y)
    
    # send the updated student model's weights to the server
    return {"student_model": serialize_model(student_model)}

if __name__ == '__main__':

    # get the client id (number) and vgg model version
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--vgg', type=int, required=True, choices=[11, 13, 16, 19])
    args = parser.parse_args()

    # store the global references
    global student_model, model, trainset, valset, testset, dataset_indices

    # student model
    student_model = vggStudent()

    # model
    if args.vgg == 11:
        model = vgg11()
    elif args.vgg == 13:
        model = vgg13()
    elif args.vgg == 16:
        model = vgg16()
    else:
        model = vgg19()
    
    # trainset
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # dataset_indices
    splitting = json.load('./splits.json')
    dataset_indices = splitting[str(args.client_id)]

    # valset, testset
    evaluation_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # split evaluation dataset into test and validation datasets
    prg = torch.Generator().manual_seed(int(os.environ["seed"]))
    valset, testset = random_split(evaluation_set, [0.5, 0.5], prg)

    # start the server
    ip = os.environ["ip"]
    port = f"{int(os.environ["base_port"]) + args.client_id + 1}"
    uvicorn.run(app, host=ip, port=port)