import json
import torch
import copy
import numpy as np
import argparse
from utils import LocalUpdate, average_weights, inference
from data_loader import load_client_config
from torch.nn import functional as F
from models.vgg import *
from models.char_cnn import *
from torchvision import datasets
from torchvision.transforms import ToTensor

def parse_data_config(path):
    with open(path) as f:
        clients_dict_lsts = json.load(f)

    clients_dict = {}
    for key in clients_dict_lsts:
        clients_dict[key] = np.array(clients_dict_lsts[key])
    
    return clients_dict

clients_dict_cifar = parse_data_config('data/cifar_config.json')
clients_dict_ag = parse_data_config('data/ag_news_config.json')

num_clients = 40

def train_isolated(args, groups, group_size, user_models, trainset, valset, dataset_name):
    # args: {gpu: boolean, optimizer: string, lr: float, local_bs: int, local_ep: int, logging: boolean, 
    # train_ratio: float}

    # user_args: {gpu: boolean, optimizer: string, lr: float, local_bs: int, local_ep: int, logging: boolean, 
    # client_id: int, clients_dict: dict, train_ratio: float}

    loss = {}
    acc = {}

    for client in range(num_clients):
        user_args = args.copy()
        user_args['client_id'] = client
        if (dataset_name == 'CIFAR10'):
            user_args['clients_dict'] = clients_dict_cifar
        elif (dataset_name == 'AG_NEWS'):
            user_args['clients_dict'] = clients_dict_ag
        else:
            raise Exception('Invalid dataset name')
    
        updater = LocalUpdate(user_args, trainset)
        model = user_models[client]
        _, l = updater.update_weights(model)

        new_loss = loss.get(groups[client], 0) + l
        loss[groups[client]] = new_loss

        acc_client, l_client = updater.inference(model, testset=valset)
        new_acc = acc.get(groups[client], 0) + acc_client
        acc[groups[client]] = new_acc
    
    for key in loss:
        loss[key] /= group_size
        acc[key] /= group_size
    
    return loss, acc

def train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
    loss = {}
    acc = {}
    weights_map = {}
    for group in groups:
        loss[group] = 0
        acc[group] = 0

    for client in range(num_clients):
        user_args = args.copy()
        if (dataset_name == 'CIFAR10'):
            user_args['clients_dict'] = clients_dict_cifar
        else:
            user_args['clients_dict'] = clients_dict_ag
        
        user_args['client_id'] = client

        updater = LocalUpdate(user_args, trainset)
        model = user_models[client]
        w, l = updater.update_weights(model)
        loss[groups[client]] += l

        acc_client, l_client = updater.inference(model, valset=valset)
        acc[groups[client]] += acc_client

        lsts = weights_map.get(groups[client], [[], []])
        client_lst = lsts[0] + [client]
        w_lst = lsts[1] + [copy.deepcopy(w)]
        weights_map[groups[client]] = [client_lst, w_lst]
    
    for group in weights_map:
        client_lst = weights_map[group][0]
        w_lst = weights_map[group][1]
        group_model = average_weights(w_lst)

        for client in client_lst:
            client_model = user_models[client]
            client_model.load_state_dict(copy.deepcopy(group_model))

    for key in loss:
        loss[key] /= group_size
        acc[key] /= group_size
    
    return loss, acc

def train_fedhat(args, groups, group_size, user_models, student_model, trainset, valset, dataset_name):
    if (dataset_name == 'CIFAR10'):
        clients_dict = clients_dict_cifar
    else:
        clients_dict = clients_dict_ag

    student_model.train()
    if (args.optimizer == 'sgd'):
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.5)

    if (args.optimizer == 'adam'):
        student_optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    loss = {}
    acc = {}
    for group in groups:
        loss[group] = 0
        acc[group] = 0

    for client in range(num_clients):
        model = user_models[client]

        trainloader, testloader = load_client_config(clients_dict, trainset, client, args.local_bs, args.train_ratio)

        if (args.optimizer == 'sgd'):
            teacher_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)

        if (args.optimizer == 'adam'):
            teacher_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        epoch_loss = []
        for epoch in range(args.local_ep):
            batch_loss = []
            for batch, (X, y) in enumerate(trainloader):
                X, y = X.to(args.device), y.to(args.device)

                pred_s = student_model(X)
                pred_t = model(X)

                task_loss_s = F.nll_loss(pred_s, y)
                task_loss_t = F.nll_loss(pred_t, y)

                distill_loss_s = F.kl_div(pred_t, pred_s) / (task_loss_s + task_loss_t)
                distill_loss_t = F.kl_div(pred_s, pred_t) / (task_loss_s + task_loss_t)

                output_t = task_loss_t + distill_loss_t
                output_s = task_loss_s + distill_loss_s

                output_t.backward()
                teacher_optimizer.step()
                teacher_optimizer.zero_grad()

                batch_loss += [copy.deepcopy(output_t.item())]

                output_s.backward()
                student_optimizer.step()
                student_optimizer.zero_grad()
            
            epoch_loss += [sum(batch_loss) / len(batch_loss)]
            if (args.logging):
                print(f"Client: {client} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")

        task_loss_fn = F.nll_loss()
        acc_client, l_client = inference(model, args.device, task_loss_fn, valset)
        loss[groups[client]] += epoch_loss[-1]
        acc[groups[client]] += acc_client

    for group in loss:
        loss[group] /= group_size
        acc[group] /= group_size


# quick tests
def main_CIFAR10():
    args = {
        'gpu': False,
        'optimizer': 'adam',
        'lr': 0.001,
        'local_bs': 32,
        'local_ep': 1,
        'logging': True,
        'train_ratio': 0.8
    }

    groups = {}
    user_models = {}
    for i in range(num_clients):
        if (i < 10):
            groups[i] = 0
            user_models[i] = vgg11()
        elif (i < 20):
            groups[i] = 1
            user_models[i] = vgg13()
        elif (i < 30):
            groups[i] = 2
            user_models[i] = vgg16()
        else:
            groups[i] = 3
            user_models[i] = vgg19()

    for i in range(10):
        #args, groups, group_size, user_models, trainset, valset, dataset_name):
        trainset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        valset = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        train_isolated(args, groups, 10, user_models, trainset, valset, dataset_name='CIFAR10')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--method', type=str, required=True)
    main_CIFAR10()