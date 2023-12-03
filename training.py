import json
import torch
import copy
import numpy as np
import argparse
import os
from utils import LocalUpdate, average_weights, inference
from data_loader import load_client_config, AGNEWS
from torch.nn import functional as F
from torch.utils.data import random_split
from models.vgg import *
from models.char_cnn import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def parse_data_config(path):
    with open(path) as f:
        clients_dict_lsts = json.load(f)

    clients_dict = {}
    for key in clients_dict_lsts:
        clients_dict[key] = np.array(clients_dict_lsts[key])
    
    return clients_dict

# global references
clients_dict_cifar = parse_data_config('data/cifar_config.json')
clients_dict_ag = parse_data_config('data/ag_news_config.json')
os.environ['num_clients'] = 40

def train_isolated(args, groups, group_size, user_models, trainset, valset, dataset_name):
    '''Isolated training subroutine (per epoch)'''
    loss = {}
    acc = {}

    for client in range(os.environ['num_clients']):
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

        acc_client, l_client = updater.inference(model, valset)
        new_acc = acc.get(groups[client], 0) + acc_client
        acc[groups[client]] = new_acc
    
    for key in loss:
        loss[key] /= group_size
        acc[key] /= group_size
    
    return loss, acc

def train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
    '''Clustered federated learning subroutine (per epoch)'''
    loss = {}
    acc = {}
    weights_map = {}

    # train individual clients
    for client in range(os.environ['num_clients']):
        user_args = args.copy()
        user_args['client_id'] = client
        if (dataset_name == 'CIFAR10'):
            user_args['client_dict'] = clients_dict_cifar
        elif (dataset_name == 'AG_NEWS'):
            user_args['client_dict'] = clients_dict_ag
        else:
            raise Exception('Invalid dataset name')

        updater = LocalUpdate(user_args, trainset)
        model = user_models[client]
        w, l = updater.update_weights(model)
        loss[groups[client]] = loss.get(groups[client], 0) + l

        acc_client, l_client = updater.inference(model, valset=valset)
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client

        lsts = weights_map.get(groups[client], [[], []])
        client_lst = lsts[0] + [client]
        w_lst = lsts[1] + [copy.deepcopy(w)]
        weights_map[groups[client]] = [client_lst, w_lst]
    
    # aggregate weights for each cluster
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
    '''FedHAT training subroutine (per epoch)'''
    if (dataset_name == 'CIFAR10'):
        clients_dict = clients_dict_cifar
    elif (dataset_name == 'AG_NEWS'):
        clients_dict = clients_dict_ag
    else:
        raise Exception('Invalid dataset name')

    student_model.train()
    if (args['optimizer'] == 'sgd'):
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.5)

    if (args['optimizer'] == 'adam'):
        student_optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    loss = {}
    acc = {}

    for client in range(os.environ['num_clients']):
        model = user_models[client]

        trainloader = load_client_config(clients_dict, trainset, client, args.local_bs)

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

                task_loss_s = F.cross_entropy(pred_s, y)
                task_loss_t = F.cross_entropy(pred_t, y)

                distill_loss_s = F.kl_div(pred_t, pred_s) / (task_loss_s + task_loss_t)
                distill_loss_t = F.kl_div(pred_s, pred_t) / (task_loss_s + task_loss_t)

                output_t = task_loss_t + distill_loss_t
                output_s = task_loss_s + distill_loss_s

                output_t.backward()
                teacher_optimizer.step()
                teacher_optimizer.zero_grad()

                output_s.backward()
                student_optimizer.step()
                student_optimizer.zero_grad()

                batch_loss += [copy.deepcopy(output_t.item())]
            
            epoch_loss += [sum(batch_loss) / len(batch_loss)]
            if (args['logging']):
                print(f"Client: {client} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")

        task_loss_fn = F.cross_entropy
        if (args['gpu'] == 'cuda'):
            device = 'cuda'
        else:
            device = 'cpu'
        acc_client, l_client = inference(model, device, task_loss_fn, valset)
        loss[groups[client]] = loss.get(groups[client], 0) + epoch_loss[-1]
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client

    for group in loss:
        loss[group] /= group_size
        acc[group] /= group_size

def create_groups_vgg():
    groups = {}
    user_models = {}

    num_clients = os.environ['num_clients']
    clients_per_model = num_clients / 4
    for i in range(os.environ['num_clients']):
        if (i < 1 * (clients_per_model)):
            groups[i] = 0
            user_models[i] = vgg11()
        elif (i < 2 * (clients_per_model)):
            groups[i] = 1
            user_models[i] = vgg13()
        elif (i < 3 * (clients_per_model)):
            groups[i] = 2
            user_models[i] = vgg16()
        else:
            groups[i] = 3
            user_models[i] = vgg19()
    
    return groups, user_models

def create_groups_char_cnn(in_channels):
    groups = {}
    user_models = {}

    num_clients = os.environ['num_clients']
    clients_per_model = num_clients / 3
    for i in range(os.environ['num_clients']):
        if (i < 1 * (clients_per_model)):
            groups[i] = 0
            user_models[i] = CharCNN(in_channels, 256, dropout=0.2)
        elif (i < 2 * (clients_per_model)):
            groups[i] = 1
            user_models[i] = CharCNN(in_channels, 512, dropout=0.2)
        else:
            groups[i] = 2
            user_models[i] = CharCNN(in_channels, 1024, dropout=0.2)
    
    return groups, user_models

def main_CIFAR10(epochs, method):
    args = {
        'gpu': False,
        'optimizer': 'adam',
        'lr': 0.001,
        'local_bs': 32,
        'local_ep': 5,
        'logging': True
    }

    groups, user_models = create_groups_vgg()
    trainset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_and_val_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    gen = torch.Generator().manual_seed(100)
    val_set, test_set = random_split(test_and_val_set, [0.5, 0.5], gen)

    for i in range(epochs):
        if (method == 'isolated'):
            train_isolated(args, groups, os.environ['num_clients'] / 4, user_models, 
                           trainset, val_set, 'CIFAR10')
        elif (method == 'clustered'):
            # train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
            train_clustered(args, groups, os.environ['num_clients'] / 4, user_models,
                            trainset, val_set, 'CIFAR10')
        elif (method == 'fedhat'):
            student_model = vggStudent()
            # train_fedhat(args, groups, group_size, user_models, student_model, trainset, valset, dataset_name):
            train_fedhat(args, groups, os.environ['num_clients'] / 4, user_models, student_model,
                         trainset, val_set, 'CIFAR10') 
        else:
            raise Exception('Invalid method provided')
    
    # testing
    print("Evaluation: ")
    acc = {}
    loss = {}
    if (args['gpu'] == 'cuda'):
        device = 'cuda'
    else:
        device = 'cpu'
    for client in range(os.environ['num_clients']):
        a, l = inference(user_models[client], device, F.cross_entropy, 
                         DataLoader(test_set, batch_size=len(test_set)//10, shuffle=False))
        acc[groups[client]] = acc.get(groups[client], 0) + a
        loss[groups[client]] = loss.get(groups[client], 0) + l
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")


def main_CIFAR10(epochs, method):
    args = {
        'gpu': False,
        'optimizer': 'adam',
        'lr': 0.001,
        'local_bs': 32,
        'local_ep': 5,
        'logging': True
    }

    train_path = 'data/ag_train.csv'
    test_path = 'data/ag_test.csv'
    alphabet_path = 'data/alphabet.json'
    trainset = AGNEWS(train_path, alphabet_path)
    test_and_val_set = AGNEWS(test_path, alphabet_path)
    in_channels = trainset.alphabet_size

    groups, user_models = create_groups_char_cnn(in_channels)

    gen = torch.Generator().manual_seed(100)
    val_set, test_set = random_split(test_and_val_set, [0.5, 0.5], gen)

    for i in range(epochs):
        if (method == 'isolated'):
            train_isolated(args, groups, os.environ['num_clients'] / 4, user_models, 
                           trainset, val_set, 'AG_NEWS')
        elif (method == 'clustered'):
            # train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
            train_clustered(args, groups, os.environ['num_clients'] / 4, user_models,
                            trainset, val_set, 'AG_NEWS')
        elif (method == 'fedhat'):
            student_model = CharCNN(in_channels, 128, dropout=0.1)
            # train_fedhat(args, groups, group_size, user_models, student_model, trainset, valset, dataset_name):
            train_fedhat(args, groups, os.environ['num_clients'] / 4, user_models, student_model,
                         trainset, val_set, 'AG_NEWS') 
        else:
            raise Exception('Invalid method provided')
    
    # testing
    print("Evaluation: ")
    acc = {}
    loss = {}
    if (args['gpu'] == 'cuda'):
        device = 'cuda'
    else:
        device = 'cpu'
    for client in range(os.environ['num_clients']):
        a, l = inference(user_models[client], device, F.cross_entropy, 
                         DataLoader(test_set, batch_size=len(test_set)//10, shuffle=False))
        acc[groups[client]] = acc.get(groups[client], 0) + a
        loss[groups[client]] = loss.get(groups[client], 0) + l
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

def main_AG_NEWS():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)

    args = parser.parse_args()
    if (args.dataset == 'CIFAR10'):
        main_CIFAR10(40, args.method)
    elif (args.dataset == 'AG_NEWS'):
        main_AG_NEWS(40, args.method)
    else:
        raise Exception('Invalid dataset')