from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import pdb

import os

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # self.network = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 512))
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128))
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512))
      #
        #for layer in self.encoder:
        #    if layer.__class__.__name__ == 'Linear':
        #    	layer.weight.data.fill_(0)

        #for layer in self.decoder:
              #    if layer.__class__.__name__ == 'Linear':
              #        layer.weight.data.fill_(0)
    

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #id = nn.Identity(x.size())
        #x = self.network(x)
        return x

def train_mapping(inputs, targets, test_inputs):
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-4

    model = autoencoder().cuda()
    criterion1 = nn.MSELoss()
    criterion2 = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset)

    print("######### Starting training of mapping function #########")

    for epoch in range(num_epochs):
        for data in dataloader:
            inp, target = data
            inp = Variable(inp).cuda()
            target = Variable(target).cuda()

            output = model(inp)
            #loss1 = criterion1(output, target)
            loss2 = criterion2(output, target, torch.ones(inp.shape[0]).cuda())
            loss = loss2
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.10f}'
            .format(epoch + 1, num_epochs, loss.data.item()))
    
    with torch.no_grad():
    	  test_inputs = torch.Tensor(test_inputs).cuda()
    	  test_outputs = model(test_inputs)
    
    return test_outputs.cpu().detach()
