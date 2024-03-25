import uvicorn
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from serialization import *
from models.vgg import *
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from typing import Dict, Any
from utils.loading import DatasetSplit
from utils.training import distill
import os
import torch
import json

class Request(BaseModel):
    student_model: str # base 64 encoding of the model
    args: Dict[str, Any]

app = FastAPI()

@app.get("/rpc")
async def fl_round(request: Request):
    args = request.args
    device = args["device"]
    student_model_weights = deserialize_model(request.student_model, device)
    student_model.load_state_dict(student_model_weights)

    # start of training
    student_model.to(device)
    student_model.train()
    model.train()
    student_optimizer = torch.optim.SGD(
        student_model.parameters(), 
        args["s_lr"], 
        momentum=args["s_momentum"], 
        weight_decay=args["s_weight_decay"]
    )
    model_optimizer = torch.optim.SGD(
        student_model.parameters(), 
        args["lr"], 
        momentum=args["momentum"], 
        weight_decay=args["weight_decay"]
    )
    trainloader = DataLoader(DatasetSplit(trainset, dataset_indices), batch_size=args["bs"], shuffle=True)

    for _ in range(args["local_ep"]):
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

    student_model = vggStudent()

    if args.vgg == 11:
        model = vgg11()
    elif args.vgg == 13:
        model = vgg13()
    elif args.vgg == 16:
        model = vgg16()
    else:
        model = vgg19()

    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    splitting = json.load('./splits.json')
    dataset_indices = splitting[str(args.client_id)]

    evaluation_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # split evaluation dataset into test and validation datasets
    prg = torch.Generator().manual_seed(int(os.environ['seed']))
    valset, testset = random_split(evaluation_set, [0.5, 0.5], prg)

    # start the server
    uvicorn.run(app, host=os.environ['ip'], port=os.environ['base_port']+args.client_id+1)