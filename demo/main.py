import argparse
import os
from utils.loading import create_client_config, load_client_dicts_as_json, create_groups_vgg
from torchvision import datasets
from torchvision.transforms import transforms
from multiprocessing import Process
import subprocess
from typing import List

def run_script(script_name: str, args: List[str]) -> None:
    """Function to run a script using subprocess"""
    subprocess.run(["python", script_name] + args, check=True)

if __name__ == '__main__':
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

    # add all of the arguments as environment variables, so
    # they can be accessed from both the server and clients
    for arg, value in vars(args).items():
        os.environ[arg] = value
    
    # load CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # allocate a non-iid distribution of data to the clients (in allocations.json)
    client_dicts = create_client_config(
        int(os.environ["num_clients"]),
        trainset,
        "CIFAR10",
    )
    load_client_dicts_as_json('./allocations.json', client_dicts)

    # assign VGG versions to each of the clients
    _, _, vgg_versions = create_groups_vgg(int(os.environ['num_clients']))

    processes = []

    # launch the server
    # processes = [Process(target=run_script, args=(script, args)) for script, args in scripts]
    processes.append(
        Process(target=run_script, args=('server.py', []))
    )
       
    # launch all of the clients
    for client_id in range(int(os.environ['num_clients'])):
        command_line_args = [
            '--client_id', client_id,
            '--vgg', vgg_versions[client_id]
        ]
        processes.append(
            Process(target=run_script, args=('client.py', command_line_args))
        )
    
    for p in processes:
        p.start()