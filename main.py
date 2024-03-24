import torch
import copy
import argparse
import os
import json
from utils.training import *
from loading import *
from torch.nn import functional as F
from models.vgg import *
from models.char_cnn import *
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from types import Dict, List, Tuple
import numpy as np

############################ TRAINING SUBROUTINES ############################

def train_isolated(
        groups: Dict[int, int], 
        group_sizes: Dict[int, int], 
        user_models: Dict[int, nn.Module],
        trainset: Dataset, 
        valset: Dataset, 
        clients_dict: Dict[int, np.ndarray[int]],
        bs: int,
        device: str,
        ep: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        logging: bool,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    '''Isolated training subroutine (per epoch); returns the average
    training loss, validation loss, and accuracy for each group.

    Arguments:
    groups: maps clients to group numbers
    group_sizes: stores sizes of each group
    user_models: maps clients to models
    trainset: training dataset
    valset: validation dataset
    clients_dict: maps clients to indices into the training dataset
    device, bs, ep, lr, momentum, weight_decay: hyperparameters for training
    '''

    # stores the training loss, validation loss, and accuracy for each client
    t_loss = {}
    v_loss = {}
    acc = {}

    for client_id in range(len(groups)):
        training_loader = load_client_loader(clients_dict, trainset, client_id, bs)
        val_loader = DataLoader(valset, batch_size=int(len(valset)/10), shuffle=False)

        _, t_l = update_weights(
            user_models[client_id],
            training_loader,
            client_id,
            device,
            ep,
            lr,
            momentum,
            weight_decay,
            logging
        )

        acc_client, v_l = inference(
            user_models[client_id],
            device,
            nn.CrossEntropyLoss(),
            val_loader,
        )

        if (logging):
            print(f"Client {client_id} / Validation Loss: {v_l} / Accuracy: {acc_client}")

        # update losses and accuracy
        t_loss[groups[client_id]] = t_loss.get(groups[client_id], 0) + t_l
        v_loss[groups[client_id]] = v_loss.get(groups[client_id], 0) + v_l
        acc[groups[client_id]] = acc.get(groups[client_id], 0) + acc_client
    
    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def train_clustered(
        groups: Dict[int, int], 
        group_sizes: Dict[int, int], 
        user_models: Dict[int, nn.Module],
        trainset: Dataset, 
        valset: Dataset, 
        clients_dict: Dict[int, np.ndarray[int]],
        bs: int,
        device: str,
        ep: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        logging: bool,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    '''Clustered federated learning subroutine (per epoch); returns the average
    training loss, validation loss, and accuracy for each group.

    Arguments:
    groups: maps clients to group numbers
    group_sizes: stores sizes of each group
    user_models: maps clients to models
    trainset: training dataset
    valset: validation dataset
    clients_dict: maps clients to indices into the training dataset
    device, bs, ep, lr, momentum, weight_decay: hyperparameters for training
    '''

    # stores the training loss, validation loss, accuracy, and weights for each group
    t_loss = {}
    v_loss = {}
    acc = {}
    weights_map = {}

    for client_id in range(len(groups)):
        training_loader = load_client_loader(clients_dict, trainset, client_id, bs)
        val_loader = DataLoader(valset, batch_size=int(len(valset)/10), shuffle=False)

        model_weights, t_l = update_weights(
            user_models[client_id],
            training_loader,
            client_id,
            device,
            ep,
            lr,
            momentum,
            weight_decay,
            logging
        )

        acc_client, v_l = inference(
            user_models[client_id],
            device,
            nn.CrossEntropyLoss(),
            val_loader,
        )

        if (logging):
            print(f"Client {client_id} / Validation Loss: {v_l} / Accuracy: {acc_client}")

        # update losses and accuracies
        t_loss[groups[client]] = t_loss.get(groups[client], 0) + t_l
        v_loss[groups[client]] = v_loss.get(groups[client], 0) + v_l
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client

        # weights map
        lsts = weights_map.get(groups[client], [[], []])
        client_lst = lsts[0] + [client]
        w_lst = lsts[1] + [copy.deepcopy(model_weights)]
        weights_map[groups[client]] = [client_lst, w_lst]
    
    # average weights for each group and update
    # each client's model
    for group in weights_map:
        client_lst = weights_map[group][0]
        w_lst = weights_map[group][1]
        group_model = average_weights(w_lst)

        for client in client_lst:
            client_model = user_models[client]
            client_model.load_state_dict(copy.deepcopy(group_model))
            user_models[client] = client_model

    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def train_hfedkd(
        groups: Dict[int, int], 
        group_sizes: Dict[int, int], 
        user_models: Dict[int, nn.Module],
        student_model: nn.Module,
        trainset: Dataset, 
        valset: Dataset, 
        clients_dict: Dict[int, np.ndarray[int]],
        bs: int,
        device: str,
        ep: int,
        lr: float,
        s_lr: float,
        momentum: float,
        s_momentum: float,
        weight_decay: float,
        s_weight_decay: float,
        logging: bool,
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    '''HFedKD training subroutine (per epoch); returns training loss, 
    validation loss, and accuracy

    Arguments:
    groups: maps clients to group numbers
    group_sizes: stores sizes of each group
    user_models: maps clients to models
    student_model: student model used for knowledge distillation
    trainset: training dataset
    valset: validation dataset
    clients_dict: maps clients to indices into the training dataset
    device, bs, ep, lr, s_lr, 
    momentum, s_momentum, weight_decay, s_weight_decay: hyperparameters for training
    '''
    student_model.train()
    student_model.to(device)
    student_optimizer = torch.optim.SGD(
        student_model.parameters(), 
        s_lr, 
        momentum=s_momentum, 
        weight_decay=s_weight_decay
    )
    
    t_loss = {}
    v_loss = {}
    acc = {}

    for client_id in range(len(groups)):
        teacher_model = user_models[client_id]
        teacher_model.to(device)
        trainloader = load_client_loader(clients_dict, trainset, client_id, bs)

        teacher_optimizer = torch.optim.SGD(
            teacher_model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )

        epoch_loss = []
        for epoch in range(ep):
            batch_loss = []
            for _, (X, y) in enumerate(trainloader):
                X, y = X.to(device), y.to(device)
                
                loss_teacher = distill(teacher_optimizer, student_model, teacher_model, X, y)
                distill(student_optimizer, teacher_model, student_model, X, y)

                batch_loss += [loss_teacher]
            
            epoch_loss += [sum(batch_loss) / len(batch_loss)]
            if (args['logging']):
                print(f"Client: {client_id} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")
        
        # inference
        valset_loader = DataLoader(valset, batch_size=int(len(valset)/10), shuffle=False)
        acc_client, v_l = inference(teacher_model, device, F.cross_entropy, valset_loader)
        
        if (logging):
            print(f"Client: {client_id} / Validation Loss: {v_l} / Accuracy: {acc_client}")

        # update losses and accuracies
        t_loss[groups[client_id]] = t_loss.get(groups[client_id], 0) + epoch_loss[-1]
        v_loss[groups[client_id]] = v_loss.get(groups[client_id], 0) + v_l
        acc[groups[client_id]] = acc.get(groups[client_id], 0) + acc_client

    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def write_outputs(
        args: argparse.Namespace, 
        t_loss: Dict[int, Dict[int, float]], 
        v_loss: Dict[int, Dict[int, float]], 
        acc_training: Dict[int, Dict[int, float]], 
        acc: Dict[int, float], 
        loss: Dict[int, float],
    ) -> None:
    '''Writes traing progress and final results to the "outputs" folder.
    
    Arguments:
    args: command line arguments
    t_loss: training loss for each group across different epochs
    v_loss validation loss for each group across different epochs
    acc_training: accuracy for each group across different epochs
    acc: accuracy for each group (evaluated on the test set)
    loss: loss for each group (evaluated on the test test)
    '''
    
    with open(f"outputs/{args['dataset']}/{args['method']}/t_loss.json", 'w') as file:
        json.dump(t_loss, file)

    with open(f"outputs/{args['dataset']}/{args['method']}/v_loss.json", 'w') as file:
        json.dump(v_loss, file)

    with open(f"outputs/{args['dataset']}/{args['method']}/acc_training.json", 'w') as file:
        json.dump(acc_training, file)
    
    with open(f"outputs/{args['dataset']}/{args['method']}/acc.json", 'w') as file:
        json.dump(acc, file)

    with open(f"outputs/{args['dataset']}/{args['method']}/loss.json", 'w') as file:
        json.dump(loss, file)


############################ TRAINING LOOPS ############################

def prepare_CIFAR10(
        args: argparse.Namespace
    ) -> Tuple[Dict, Dict, Dict, Dict, Dataset, Dataset, Dataset]:

    '''Allocates models and training samples to clients and returns training,
    validation, test datasets for the CIFAR10 dataset.'''

    # assign models to clients
    groups, group_sizes, user_models = create_groups_vgg(args.num_clients)

    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    evaluation_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # allocate training samples to clients in a non-iid fashion
    clients_dict = create_client_config(args.num_clients, trainset, 'CIFAR10')

    # split evaluation dataset into test and validation datasets
    prg = torch.Generator().manual_seed(int(os.environ['seed']))
    valset, testset = random_split(evaluation_set, [0.5, 0.5], prg)

    return groups, group_sizes, user_models, clients_dict, trainset, valset, testset

def prepare_AG_NEWS(
        args: argparse.Namespace
    ) -> Tuple[Dict, Dict, Dict, Dict, Dataset, Dataset, Dataset, int]:

    '''Allocates models and training samples to clients and returns training,
    validation, test datasets for the AG_NEWS dataset.'''

    # load AG News training and evaluation datasets
    trainset = AGNEWS('data/ag_train.csv', 'data/alphabet.json')
    evaluation_set = AGNEWS('data/ag_test.csv', 'data/alphabet.json')
    in_channels = trainset.alphabet_size

    # assign models to clients
    clients_dict = create_client_config(args.num_clients, trainset, 'AG_NEWS')
    groups, group_sizes, user_models = create_groups_char_cnn(args.num_clients, in_channels)

    prg = torch.Generator().manual_seed(int(os.environ['seed']))
    valset, testset = random_split(evaluation_set, [0.5, 0.5], prg)

    return groups, group_sizes, user_models, clients_dict, trainset, valset, testset, in_channels

def main(
        args: argparse.Namespace, 
        groups: Dict[int, int], 
        group_sizes: Dict[int, int], 
        user_models: Dict[int, nn.Module], 
        clients_dict: Dict[int, np.ndarray[int]], 
        trainset: Dataset, 
        valset: Dataset, 
        testset: Dataset, 
        in_channels=None
    ) -> None:
    '''Abstract training loop'''
    
    t_loss = {}
    v_loss = {}
    acc_training = {}

    student_model = None
    if (args.method == 'hfedkd'):
        if args.dataset == 'CIFAR10':
            student_model = CharCNN(in_channels, 128, dropout=0.1)
        else:
            student_model = vggStudent()

    for i in range(args.epochs):
        if (args.method == 'isolated'):
            t_loss[i], v_loss[i], acc_training[i] = train_isolated(
                groups,
                group_sizes,
                user_models,
                trainset,
                valset,
                clients_dict,
                args.bs,
                args.device,
                args.local_ep,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.logging,
            )
        elif (args.method == 'clustered'):
            t_loss[i], v_loss[i], acc_training[i] = train_clustered(
                groups,
                group_sizes,
                user_models,
                trainset,
                valset,
                clients_dict,
                args.bs,
                args.device,
                args.local_ep,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.logging,
            )
        elif (args.method == 'hfedkd'):
            student_model = vggStudent()
            t_loss[i], v_loss[i], acc_training[i] = train_hfedkd(
                groups,
                group_sizes,
                user_models,
                student_model,
                trainset,
                valset,
                clients_dict,
                args.bs,
                args.device,
                args.local_ep,
                args.lr,
                args.s_lr,
                args.momentum,
                args.s_momentum,
                args.weight_decay,
                args.s_weight_decay,
                args.logging,
            )
        else:
            raise Exception('Invalid method provided')
    
    # evaluation
    acc = {}
    loss = {}
    for client_id in range(args.num_clients):
        a, l = inference(
            user_models[client_id],
            args.device,
            F.cross_entropy,
            DataLoader(testset, batch_size=len(testset)//10, shuffle=False)
        )
        acc[groups[client_id]] = acc.get(groups[client_id], 0) + a
        loss[groups[client_id]] = loss.get(groups[client_id], 0) + l

    # average accuracies and losses for groups
    for group in loss:
        loss[group] /= group_sizes[group]
        acc[group] /= group_sizes[group]
    
    # store intermediate and final results
    write_outputs(args, t_loss, v_loss, acc_training, acc, loss)
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

if __name__ == '__main__':
    os.environ['seed'] = "100"
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_clients', type=int, default=24)
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'AG_NEWS'])
    parser.add_argument('--method', type=str, required=True, choices=['isolated', 'clustered', 'hfedkd'])
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
    parser.add_argument('--logging', type=bool, default=False)
    
    args = parser.parse_args()

    if (args.dataset == 'CIFAR10'):
        main(prepare_CIFAR10(args))
    
    if (args.dataset == 'AG_NEWS'):
        main(prepare_AG_NEWS(args))