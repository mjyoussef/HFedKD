import httpx
import asyncio
import argparse
import torch.nn as nn
from models.vgg import vggStudent
from serialization import *
from utils.training import average_weights
import os

async def make_rpc_call(
        args: argparse.Namespace, 
        client_id: int,
        student_model: nn.Module,
    ) -> nn.Module:

    url = f"http://{os.environ['ip']}:{os.environ['base_port'] + client_id + 1}/rpc"

    client_args = {
        "device": args.device,
        "lr": args.lr,
        "s_lr": args.s_lr,
        "momentum": args.momentum,
        "s_momentum": args.s_momentum,
        "weight_decay": args.weight_decay,
        "s_weight_decay": args.s_weight_decay,
        "local_ep": args.local_ep,
        "bs": args.bs,
    }

    async with httpx.AsyncClient() as client:
        # serialize and send to the client
        data_b64 = serialize_model(student_model)
        response = await client.post(url, json={"student_model": data_b64, "args": client_args})
        
        # make sure the response was successful
        response.raise_for_status()
        
        # update the list of student models (in the future,
        # we'll pass these into FedAvg and use the resulting student model
        # in future rounds of FL)
        response_data = await response.json()
        student_model_weights = deserialize_model(response_data.get("student_model"), args.device)
        return student_model_weights
    
async def main(args):
    student_model = vggStudent()

    for epoch in range(args.epochs):
        calls = []
        for client_id in range(args.num_clients):
            calls.append(make_rpc_call(args, client_id, student_model))
        
        # wait on all of the calls
        updated_student_models = await asyncio.gather(*calls)

        # average the weights and load into the student model
        avg_student_model = average_weights(updated_student_models)
        student_model.load_state_dict(avg_student_model)

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=24)
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--s_lr', type=float, required=False, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--s_weight_decay', type=float, required=False, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--s_momentum', type=float, required=False, default=0.9)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--local_ep', type=int, default=7)
    args = parser.parse_args()

    # training
    main(args)