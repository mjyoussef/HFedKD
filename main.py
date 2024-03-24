import torch
import copy
import argparse
import os
import json
from utils.training import LocalUpdate, average_weights, inference
from utils.data_loader import load_client_config, AGNEWS, create_client_config
from torch.nn import functional as F
from models.vgg import *
from models.char_cnn import *
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

# global references
clients_dict_cifar = None
clients_dict_ag = None
os.environ['seed'] = "100"

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
        params, t_l = updater.update_weights(model)
        acc_client, v_l = updater.inference(model, valset)
        if (args['logging']):
            print(f"Client {client} / Validation Loss: {v_l} / Accuracy: {acc_client}")

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
            user_args['clients_dict'] = clients_dict_cifar
        elif (dataset_name == 'AG_NEWS'):
            user_args['clients_dict'] = clients_dict_ag
        else:
            raise Exception('Invalid dataset name')

        updater = LocalUpdate(user_args, trainset)
        model = user_models[client]
        w, t_l = updater.update_weights(model)
        acc_client, v_l = updater.inference(model, valset)
        if (args['logging']):
            print(f"Client {client} / Validation Loss: {v_l} / Accuracy: {acc_client}")

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
            user_models[client] = client_model

    # average losses and accuracies for each group
    for key in t_loss:
        t_loss[key] /= group_sizes[key]
        v_loss[key] /= group_sizes[key]
        acc[key] /= group_sizes[key]
    
    return t_loss, v_loss, acc

def distill(optimizer, target, source, X, y, weights=[0.8, 0.2], temp=2):
    '''Knowledge distillation from source to target.
    
    Note: a default temperature (`temp`) = 2 is used in the softmax object'''
    optimizer.zero_grad()
    
    target.temp, source.temp = temp, temp
    
    with torch.no_grad():
        source_out = source(X)
    
    target_out = target(X)
    
    target_task_loss = F.cross_entropy(target_out, y)
    
    distill_loss = F.kl_div(source_out, target_out)
    
    total_loss = (weights[0] * target_task_loss) + (weights[1] * distill_loss)

    total_loss.backward()
    optimizer.step()
    
    target.temp, source.temp = None, None
    return total_loss.item()

def train_fedhat(args, groups, group_sizes, user_models, student_model, trainset, valset, dataset_name):
    '''HFedKD training subroutine (per epoch)'''
    if (dataset_name == 'CIFAR10'):
        clients_dict = clients_dict_cifar
    elif (dataset_name == 'AG_NEWS'):
        clients_dict = clients_dict_ag
    else:
        raise Exception('Invalid dataset name')
    
    device = args['device']

    student_model.train()
    student_model.to(device)
    student_optimizer = torch.optim.SGD(student_model.parameters(), 0.001, momentum=0.9)
    
    t_loss = {}
    v_loss = {}
    acc = {}

    for client in range(args['num_clients']):
        model = user_models[client]
        model.to(device)
        trainloader = load_client_config(clients_dict, trainset, client, args['local_bs'])

        if (args['optimizer'] == 'sgd'):
            teacher_optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=1e-4)

        if (args['optimizer'] == 'adam'):
            teacher_optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

        epoch_loss = []
        for epoch in range(args['local_ep']):
            batch_loss = []
            for batch, (X, y) in enumerate(trainloader):
                X, y = X.to(device), y.to(device)
                
                loss_teacher = distill(teacher_optimizer, model, student_model, X, y)
                _ = distill(student_optimizer, student_model, model, X, y)

                batch_loss += [loss_teacher]
            
            epoch_loss += [sum(batch_loss) / len(batch_loss)]
            if (args['logging']):
                print(f"Client: {client} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")
        
        # inference
        task_loss_fn = F.cross_entropy
        valset_loader = DataLoader(valset, batch_size=int(len(valset)/10), shuffle=False)
        acc_client, v_l = inference(model, device, task_loss_fn, valset_loader)
        
        if (args['logging']):
            print(f"Client: {client} / Validation Loss: {v_l} / Accuracy: {acc_client}")

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

def write_outputs(args, t_loss, v_loss, acc_training, acc, loss):
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

    global clients_dict_ag
    clients_dict_ag = create_client_config(args['num_clients'], trainset, 'CIFAR10')

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
    write_outputs(args, t_loss, v_loss, acc_training, acc, loss)
    
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

    global clients_dict_ag
    clients_dict_ag = create_client_config(args['num_clients'], trainset, 'AG_NEWS')

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
    write_outputs(args, t_loss, v_loss, acc_training, acc, loss)
    
    print(f"Accuracy: {acc}")
    print(f"Loss: {loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_clients', type=int, default=24)
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'AG_NEWS'])
    parser.add_argument('--method', type=str, required=True, choices=['isolated', 'clustered', 'fedhat'])
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--local_ep', type=int, default=7)
    parser.add_argument('--logging', type=bool, default=False)
    
    args = parser.parse_args()

    if (args.dataset == 'CIFAR10'):
        main_CIFAR10(vars(args))
    
    if (args.dataset == 'AG_NEWS'):
        main_AG_NEWS(vars(args))