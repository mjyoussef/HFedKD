import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.data_loader import load_client_config
import copy

class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.client_id = self.args['client_id']
        self.trainloader = load_client_config(self.args['clients_dict'], dataset, 
                                              self.args['client_id'], self.args['local_bs'])
        self.device = 'cuda' if self.args['gpu'] else 'cpu'
        self.loss = nn.CrossEntropyLoss().to(self.device)
    
    def update_weights(self, model):
        model.train(True)

        if self.args['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args['lr'], momentum=0.5)
        if self.args['optimizer']  == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args['lr'], weight_decay=1e-4)
            
        epoch_loss = []
        for epoch in range(self.args['local_ep']):
            batch_loss = []
            for batch, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)

                pred = model(X)
                loss = self.loss(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_loss += [loss.item()]
            
            epoch_loss += [sum(batch_loss) / len(batch_loss)]
            if (self.args['logging']):
                print(f"Client: {self.client_id} / Epoch: {epoch} / Loss: {epoch_loss[-1]}")
        
        return model.state_dict(), epoch_loss[-1]
            
    def inference(self, model, testset):
        testloader = DataLoader(testset, batch_size=int(len(testset)/10), shuffle=False)
        return inference(model, self.device, self.loss, testloader)

def inference(model, device, loss, testloader):
    model.eval()
    batch_loss, total, correct = [], 0.0, 0.0

    with torch.no_grad():
        for batch, (X, y) in enumerate(testloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            l = loss(pred, y)
            batch_loss += [l.item()]

            _, pred_labels = torch.max(pred, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, y)).item()
            total += len(y)
    
    return correct / total, sum(batch_loss) / len(batch_loss)
    
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg