import torch
import copy
import argparse
import os
import json
from utils.training import LocalUpdate, average_weights, inference
from utils.data_loader import load_client_config, AGNEWS, parse_data_config
from torch.nn import functional as F
from models.vgg import *
from models.char_cnn import *
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

############################ TRAINING SUBROUTINES ############################

def train_isolated(args, groups, group_sizes, user_models, trainset, valset, dataset_name):
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
        del updater, model_params

        # training loss
        t_loss[groups[client]] = t_loss.get(groups[client], 0) + t_l

        # validation loss
        v_loss[groups[client]] = v_loss.get(groups[client], 0) + v_l

        # accuracy
        acc[groups[client]] = acc.get(groups[client], 0) + acc_client
    
    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def train_clustered(args, groups, group_sizes, user_models, trainset, valset, dataset_name):
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
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def train_fedhat(args, groups, group_sizes, user_models, student_model, trainset, valset, dataset_name):
    '''FedHAT training subroutine (per epoch)'''
    if (dataset_name == 'CIFAR10'):
        clients_dict = clients_dict_cifar
    elif (dataset_name == 'AG_NEWS'):
        clients_dict = clients_dict_ag
    else:
        raise Exception('Invalid dataset name')
    
    device = args['device']

    student_model.train()
    student_model.to(device)
    if (args['optimizer'] == 'sgd'):
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

    if (args['optimizer'] == 'adam'):
        student_optimizer = torch.optim.Adam(student_model.parameters(), lr=args['lr'], weight_decay=1e-4)
    
    t_loss = {}
    v_loss = {}
    acc = {}

    for client in range(args['num_clients']):
        model = user_models[client]
        model.to(device)
        trainloader = load_client_config(clients_dict, trainset, client, args['local_bs'])

        if (args['optimizer'] == 'sgd'):
            teacher_optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

        if (args['optimizer'] == 'adam'):
            teacher_optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

        epoch_loss = []
        for epoch in range(args['local_ep']):
            batch_loss = []
            for batch, (X, y) in enumerate(trainloader):
                X, y = X.to(device), y.to(device)

                teacher_optimizer.zero_grad()
                student_optimizer.zero_grad()

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

                output_s.backward()
                student_optimizer.step()

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
    for key in t_loss:
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def create_groups_vgg(num_clients):
    groups = {}
    group_sizes = {}
    user_models = {}

    clients_per_model = num_clients // 4
    counter, group = 0, -1
    rem = num_clients % 4
    for i in range(num_clients):
        if (counter == 0):
            counter = clients_per_model
            if (rem > 0):
                counter += 1
                rem -= 1
            group += 1
            group_sizes[group] = counter
        
        groups[i] = group
        
        if (group == 0):
            user_models[i] = vgg11()
        elif (group == 1):
            user_models[i] = vgg13()
        elif (group == 2):
            user_models[i] = vgg16()
        else:
            user_models[i] = vgg19()
        
        counter -= 1
    
    return groups, group_sizes, user_models

def create_groups_char_cnn(num_clients, in_channels):
    groups = {}
    group_sizes = {}
    user_models = {}

    clients_per_model = num_clients // 3
    counter, group = 0, -1
    rem = num_clients % 3
    for i in range(num_clients):
        if (counter == 0):
            counter = clients_per_model
            if (rem > 0):
                counter += 1
                rem -= 1
            group += 1
            group_sizes[group] = counter
        
        groups[i] = group
        
        if (group == 0):
            user_models[i] = CharCNN(in_channels, 256, dropout=0.2)
        elif (group == 1):
            user_models[i] = CharCNN(in_channels, 512, dropout=0.2)
        else:
            user_models[i] = CharCNN(in_channels, 1024, dropout=0.2)
        
        counter -= 1
    
    return groups, group_sizes, user_models


############################ TRAINING LOOPS ############################

def main_CIFAR10(args):

    # assign models to clients
    groups, group_sizes, user_models = create_groups_vgg(args['num_clients'])
    t_loss, v_loss, acc_training = {}, {}, {}

    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_and_val_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    gen = torch.Generator().manual_seed(int(os.environ['seed']))
    val_set, test_set = random_split(test_and_val_set, [0.5, 0.5], gen)

    for i in range(args['epochs']):
        if (args['method'] == 'isolated'):
            t_loss[i], v_loss[i], acc_training[i] = train_isolated(args, groups, group_sizes, user_models, 
                                                          trainset, val_set, 'CIFAR10')
        elif (args['method'] == 'clustered'):
            t_loss[i], v_loss[i], acc_training[i] = train_clustered(args, groups, group_sizes, user_models,
                                                           trainset, val_set, 'CIFAR10')
        elif (args['method'] == 'fedhat'):
            student_model = vggStudent()
            t_loss[i], v_loss[i], acc_training[i] = train_fedhat(args, groups, group_sizes, user_models, student_model,
                                                        trainset, val_set, 'CIFAR10')
        else:
            raise Exception('Invalid method provided')
    
    # evaluation
    print("Evaluation: ")
    acc = {}
    loss = {}
    device = args['device']
    for client in range(args['num_clients']):
        a, l = inference(user_models[client], device, F.cross_entropy, 
                         DataLoader(test_set, batch_size=len(test_set)//10, shuffle=False))
        acc[groups[client]] = acc.get(groups[client], 0) + a
        loss[groups[client]] = loss.get(groups[client], 0) + l

    # average accuracies and losses for groups
    for group in loss:
        loss[group] /= group_sizes[group]
        acc[group] /= group_sizes[group]
    

    # store intermediate and final results
    with open(args['t_loss_path'], 'w') as file:
        json.dump(t_loss, file)

    with open(args['v_loss_path'], 'w') as file:
        json.dump(v_loss, file)

    with open(args['acc_training_path'], 'w') as file:
        json.dump(acc_training, file)
    
    with open(args['acc_path'], 'w') as file:
        json.dump(acc, file)

    with open(args['loss_path'], 'w') as file:
        json.dump(loss, file)
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

def main_AG_NEWS(args):
    train_path = 'data/ag_train.csv'
    test_path = 'data/ag_test.csv'
    alphabet_path = 'data/alphabet.json'
    trainset = AGNEWS(train_path, alphabet_path)
    test_and_val_set = AGNEWS(test_path, alphabet_path)
    in_channels = trainset.alphabet_size

    t_loss, v_loss, acc_training = {}, {}, {}

    # assign models to clients
    groups, group_sizes, user_models = create_groups_char_cnn(args['num_clients'], in_channels)

    gen = torch.Generator().manual_seed(int(os.environ['seed']))
    val_set, test_set = random_split(test_and_val_set, [0.5, 0.5], gen)

    for i in range(args['epochs']):
        if (args['method'] == 'isolated'):
            t_loss[i], v_loss[i], acc_training[i] = train_isolated(args, groups, group_sizes, user_models, 
                                                                   trainset, val_set, 'AG_NEWS')
        elif (args['method'] == 'clustered'):
            # train_clustered(args, groups, group_size, user_models, trainset, valset, dataset_name):
            t_loss[i], v_loss[i], acc_training[i] = train_clustered(args, groups, group_sizes, user_models,
                                                                    trainset, val_set, 'AG_NEWS')
        elif (args['method'] == 'fedhat'):
            student_model = CharCNN(in_channels, 128, dropout=0.1)
            # train_fedhat(args, groups, group_size, user_models, student_model, trainset, valset, dataset_name):
            t_loss[i], v_loss[i], acc_training[i] = train_fedhat(args, groups, group_sizes, user_models, student_model,
                                                                 trainset, val_set, 'AG_NEWS')
        else:
            raise Exception('Invalid method provided')
    
    # evaluation
    print("Evaluation: ")
    acc = {}
    loss = {}
    device = args['device']
    for client in range(args['num_clients']):
        a, l = inference(user_models[client], device, F.cross_entropy, 
                         DataLoader(test_set, batch_size=len(test_set)//10, shuffle=False))
        acc[groups[client]] = acc.get(groups[client], 0) + a
        loss[groups[client]] = loss.get(groups[client], 0) + l
    
    # average accuracies and losses among groups
    for group in loss:
        loss[group] /= group_sizes[group]
        acc[group] /= group_sizes[group]
    
    # store intermediate and final results
    with open(args['t_loss_path'], 'w') as file:
        json.dump(t_loss, file)

    with open(args['v_loss_path'], 'w') as file:
        json.dump(v_loss, file)

    with open(args['acc_training_path'], 'w') as file:
        json.dump(acc_training, file)
    
    with open(args['acc_path'], 'w') as file:
        json.dump(acc, file)

    with open(args['loss_path'], 'w') as file:
        json.dump(loss, file)
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hardcoded variables
    os.environ['seed'] = "100"

    # don't change unless using a custom configuration for clients' data distributions
    parser.add_argument('--num_clients', type=int, default=24)

    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'AG_NEWS'])
    parser.add_argument('--method', type=str, required=True, choices=['isolated', 'clustered', 'fedhat'])
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--local_ep', type=int, default=10)
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--t_loss_path', type=str, default='outputs/t_loss.json')
    parser.add_argument('--v_loss_path', type=str, default='outputs/v_loss.json')
    parser.add_argument('--acc_training_path', type=str, default='outputs/acc_training.json')
    parser.add_argument('--acc_path', type=str, default='outputs/acc.json')
    parser.add_argument('--loss_path', type=str, default='outputs/loss.json')
    
    args = parser.parse_args()

    clients_dict_cifar = parse_data_config('data/cifar_config.json')
    clients_dict_ag = parse_data_config('data/ag_news_config.json')

    # python3 main.py --num_clients 40 --dataset CIFAR10 --method isolated --logging True --local_ep 2 --device mps 
    # python3 main.py --num_clients 40 --dataset AG_NEWS --method isolated --logging True --local_ep 2 --device mps

    if (args.dataset == 'CIFAR10'):
        main_CIFAR10(vars(args))
    
    if (args.dataset == 'AG_NEWS'):
        main_AG_NEWS(vars(args))