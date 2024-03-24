import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import copy

def update_weights(model: nn.Module, 
                   trainloader: DataLoader, 
                   client_id: str | int, 
                   device: str, 
                   ep: int, 
                   lr: float, 
                   momentum: float, 
                   weight_decay: float, 
                   logging=True) -> Tuple[Dict[str, Any], float]:
    '''Returns the trained model parameters and epoch loss and, optionally, logs intermediate results.'''
    model.to(device)
    model.train()
    loss = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
        
    epoch_loss = []
    for epoch in range(ep):
        batch_loss = []
        for _, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            l = loss(pred, y)

            l.backward()
            optimizer.step()

            batch_loss += [l.item()]
        
        epoch_loss += [sum(batch_loss) / len(batch_loss)]
        if (logging):
            print(f"Client: {client_id} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")
            
    return model.state_dict(), epoch_loss[-1]

def average_weights(w: Dict[str, Any]) -> int:
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def distill(optimizer: torch.optim.Optimizer, 
            source: nn.Module, 
            target: nn.Module, 
            X: torch.Tensor, 
            y: torch.Tensor, 
            weights=[0.8, 0.2], 
            temperature=2) -> float:
    '''Knowledge distillation from source to target. This step only optimizes
    the target model and leaves the source model's weights intact.'''
    optimizer.zero_grad()
    
    with torch.no_grad():
        source_out = source(X, temperature)
    
    target_out = target(X, temperature)
    
    target_task_loss = F.cross_entropy(target_out, y)
    
    distill_loss = F.kl_div(source_out, target_out)
    
    total_loss = (weights[0] * target_task_loss) + (weights[1] * distill_loss)

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def inference(model: nn.Module, 
              device: str, 
              loss: nn.Module, 
              testloader: DataLoader) -> Tuple[float, float]:
    '''Evaluates the model, returning accuracy and average batch loss.'''
    model.to(device)
    model.eval()
    batch_loss, total, correct = [], 0.0, 0.0

    with torch.no_grad():
        for _, (X, y) in enumerate(testloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            l = loss(pred, y)
            batch_loss += [l.item()]

            _, pred_labels = torch.max(pred, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, y)).item()
            total += len(y)
    
    return correct / total, sum(batch_loss) / len(batch_loss)