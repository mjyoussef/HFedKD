import torch
import copy
import argparse
import os
from utils.training import LocalUpdate, average_weights, inference
from utils.data_loader import load_client_config, AGNEWS, parse_data_config
from torch.nn import functional as F
from torch.utils.data import random_split
from models.vgg import *
from models.char_cnn import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

############################ TRAINING SUBROUTINES ############################

def train_isolated(args, groups, group_size, user_models, trainset, valset, dataset_name):
    '''Isolated training subroutine (per epoch)'''
    t_loss = {}
    v_loss = {}
    acc = {}

    for client in range(args['num_clients']):
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
        model_params, t_l = updater.update_weights(model)
        acc_client, v_l = updater.inference(model, valset)

        # training loss
        t_loss[groups[client]] = t_loss.get(groups[client], 0) + t_l

        # validation loss
        v_loss[groups[client]] = v_loss.get(groups[client], 0) + v_l

        # accuracy
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client
    
    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_size
        v_loss[key] /= group_size
        acc[key] /= group_size
    
    return t_loss, v_loss, acc

def train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
    '''Clustered federated learning subroutine (per epoch)'''
    t_loss = {}
    v_loss = {}
    acc = {}
    weights_map = {}

    # train individual clients
    for client in range(args['num_clients']):
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
        w, t_l = updater.update_weights(model)
        acc_client, v_l = updater.inference(model, valset)

        # training loss
        t_loss[groups[client]] = t_loss.get(groups[client], 0) + t_l

        # validation loss
        v_loss[groups[client]] = v_loss.get(groups[client], 0) + v_l

        # accuracy
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client

        # weights map
        lsts = weights_map.get(groups[client], [[], []])
        client_lst = lsts[0] + [client]
        w_lst = lsts[1] + [copy.deepcopy(w)]
        weights_map[groups[client]] = [client_lst, w_lst]
    
    # aggregate weights for each group
    for group in weights_map:
        client_lst = weights_map[group][0]
        w_lst = weights_map[group][1]
        group_model = average_weights(w_lst)

        for client in client_lst:
            client_model = user_models[client]
            client_model.load_state_dict(copy.deepcopy(group_model))

    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_size
        v_loss[key] /= group_size
        acc[key] /= group_size
    
    return t_loss, v_loss, acc

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
    
    device = 'cuda' if args['gpu'] else 'cpu'
    
    t_loss = {}
    v_loss = {}
    acc = {}

    for client in range(args['num_clients']):
        model = user_models[client]
        trainloader = load_client_config(clients_dict, trainset, client, args['local_bs'])

        if (args['optimizer'] == 'sgd'):
            teacher_optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.5)

        if (args['optimizer'] == 'adam'):
            teacher_optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

        epoch_loss = []
        for epoch in range(args['local_ep']):
            batch_loss = []
            for batch, (X, y) in enumerate(trainloader):
                X, y = X.to(device), y.to(device)

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

                batch_loss += [output_t.item()]
            
            epoch_loss += [sum(batch_loss) / len(batch_loss)]
            if (args['logging']):
                print(f"Client: {client} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")
        
        # inference
        task_loss_fn = F.cross_entropy
        acc_client, v_l = inference(model, device, task_loss_fn, valset)

        # training loss
        t_loss[groups[client]] = t_loss.get(groups[client], 0) + epoch_loss[-1]

        # validation loss
        v_loss[groups[client]] = v_loss.get(groups[client], 0) + v_l

        # accuracy
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client

    # average losses and accuracies for each group
    for group in t_loss:
        t_loss[group] /= group_size
        v_loss[group] /= group_size
        acc[group] /= group_size
    
    return t_loss, v_loss, acc

def create_groups_vgg(num_clients):
    groups = {}
    user_models = {}

    clients_per_model = num_clients / 4
    for i in range(num_clients):
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

def create_groups_char_cnn(num_clients, in_channels):
    groups = {}
    user_models = {}

    clients_per_model = num_clients / 3
    for i in range(num_clients):
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


############################ TRAINING LOOPS ############################

def main_CIFAR10(args):

    # assign models to clients
    groups, user_models = create_groups_vgg(args['num_clients'])

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

    gen = torch.Generator().manual_seed(os.environ['seed'])
    val_set, test_set = random_split(test_and_val_set, [0.5, 0.5], gen)

    group_size = args['num_clients'] // os.environ['num_vgg_models']
    for i in range(args['epochs']):
        if (args['method'] == 'isolated'):
            train_isolated(args, groups, group_size, user_models, 
                           trainset, val_set, 'CIFAR10')
        elif (args['method'] == 'clustered'):
            train_clustered(args, groups, group_size, user_models,
                            trainset, val_set, 'CIFAR10')
        elif (args['method'] == 'fedhat'):
            student_model = vggStudent()
            train_fedhat(args, groups, group_size, user_models, student_model,
                         trainset, val_set, 'CIFAR10') 
        else:
            raise Exception('Invalid method provided')
    
    # evaluation
    print("Evaluation: ")
    acc = {}
    loss = {}
    device = 'cuda' if args['gpu'] else 'cpu'
    for client in range(args['num_clients']):
        a, l = inference(user_models[client], device, F.cross_entropy, 
                         DataLoader(test_set, batch_size=len(test_set)//10, shuffle=False))
        acc[groups[client]] = acc.get(groups[client], 0) + a
        loss[groups[client]] = loss.get(groups[client], 0) + l

    # average accuracies and losses for groups
    for group in loss:
        loss[group] /= group_size
        acc[group] /= group_size
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

def main_AG_NEWS(args):
    train_path = 'data/ag_train.csv'
    test_path = 'data/ag_test.csv'
    alphabet_path = 'data/alphabet.json'
    trainset = AGNEWS(train_path, alphabet_path)
    test_and_val_set = AGNEWS(test_path, alphabet_path)
    in_channels = trainset.alphabet_size

    # assign models to clients
    groups, user_models = create_groups_char_cnn(args['num_clients'], in_channels)

    gen = torch.Generator().manual_seed(os.environ['seed'])
    val_set, test_set = random_split(test_and_val_set, [0.5, 0.5], gen)

    group_size = args['num_clients'] // os.environ['num_charcnn_models']
    for i in range(args['epochs']):
        if (args['method'] == 'isolated'):
            train_isolated(args, groups, group_size, user_models, 
                           trainset, val_set, 'AG_NEWS')
        elif (args['method'] == 'clustered'):
            # train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
            train_clustered(args, groups, group_size, user_models,
                            trainset, val_set, 'AG_NEWS')
        elif (args['method'] == 'fedhat'):
            student_model = CharCNN(in_channels, 128, dropout=0.1)
            # train_fedhat(args, groups, group_size, user_models, student_model, trainset, valset, dataset_name):
            train_fedhat(args, groups, group_size, user_models, student_model,
                         trainset, val_set, 'AG_NEWS')
        else:
            raise Exception('Invalid method provided')
    
    # evaluation
    print("Evaluation: ")
    acc = {}
    loss = {}
    device = 'cuda' if args['gpu'] else 'cpu'
    for client in range(args['num_clients']):
        a, l = inference(user_models[client], device, F.cross_entropy, 
                         DataLoader(test_set, batch_size=len(test_set)//10, shuffle=False))
        acc[groups[client]] = acc.get(groups[client], 0) + a
        loss[groups[client]] = loss.get(groups[client], 0) + l
    
    # average accuracies and losses among groups
    for group in loss:
        loss[group] /= group_size
        acc[group] /= group_size
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hardcoded variables
    os.environ['seed'] = 100
    os.environ['num_vgg_models'] = 4
    os.environ['num_charcnn_models'] = 3

    # `num_clients` is evenly distributed among model groups
    parser.add_argument('--num_clients', type=int, required=True)

    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10, AG_NEWS'])
    parser.add_argument('--method', type=str, required=True, choices=['isolated', 'clustered', 'fedhat'])
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--local_bs', type=int, default=32)
    parser.add_argument('--local_ep', type=int, default=7)
    parser.add_argument('--logging', type=bool, default=False)
    
    args = parser.parse_args()
    if (args.dataset == 'CIFAR10' and args.num_clients % os.environ['num_vgg_models'] != 0):
        raise Exception('Number of clients is invalid')
    
    if (args.dataset == 'AG_NEWS' and args.num_clients % os.environ['num_charcnn_models'] != 0):
        raise Exception('Number of clients is invalid')

    clients_dict_cifar = parse_data_config('data/cifar_config.json')
    clients_dict_ag = parse_data_config('data/ag_news_config.json')

    if (args.dataset == 'CIFAR10'):
        main_CIFAR10(vars(args))
    
    if (args.dataset == 'AG_NEWS'):
        main_AG_NEWS(vars(args))