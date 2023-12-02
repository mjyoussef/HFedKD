import json
import copy
import numpy as np
import argparse
from utils import LocalUpdate, average_weights

def parse_data_config(path):
    clients_dict_lsts = json.load(path)
    clients_dict = {}
    for key in clients_dict_lsts:
        clients_dict[key] = np.array(clients_dict_lsts[key])

clients_dict_cifar = parse_data_config('/data/cifar_config.json')
clients_dict_ag = parse_data_config('/data/ag_news_config.json')

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
        if (dataset_name == 'CIFAR10'):
            user_args['clients_dict'] = clients_dict_cifar
        else:
            user_args['clients_dict'] = clients_dict_ag
        
        user_args['client_id'] = client

        updater = LocalUpdate(user_args, trainset)
        model = user_models[client]
        loss[groups[client]] += updater.update_weights(model)
        acc[groups[client]] += updater.inference(model, valset=valset)
    
    for key in loss:
        loss[key] /= group_size
        acc[key] /= group_size
    
    return loss, acc

def train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
    loss = {}
    acc = {}
    weights_map = {}

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
        acc[groups[client]] += updater.inference(model, valset=valset)

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

def train_fedhat():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)