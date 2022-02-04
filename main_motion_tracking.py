import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from network import *
from dataio import *
import time
from network import *
from utils import *

n_worker = 1
bs = 1
max_norm = 0.25

VAE_model_load_path = './models/main_prior_model_pretrained.pth'
z_dim = 32
factor = 8

flow_criterion = nn.MSELoss(reduction='sum')
Tensor = torch.cuda.FloatTensor

def test():

    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        x, x_pred, mask = batch
        x = x[0]
        x_pred = x_pred[0]
        n_seq = x.shape[0]

        Trans_model = Temporal_VAE(img_size=96, z_dim=z_dim, factor=factor, n_seq=n_seq)
        Trans_model = Trans_model.cuda()
        Trans_model.load_state_dict(torch.load(VAE_model_load_path))

        x_c = Variable(x.type(Tensor))
        x_predc = Variable(x_pred.type(Tensor))           
        mask = Variable(mask.type(Tensor))

        z_recall = torch.tensor(np.random.normal(0.0, 0.8, (n_seq, z_dim)), dtype=torch.float, requires_grad=True, device="cuda:0")

        optimizer = optim.Adam([{'params': z_recall}], lr=0.1, weight_decay=1e-3)

        weight = define_weight(length=n_seq)
        weight = torch.tensor(weight, dtype=torch.float, device="cuda:0")

        delta_loss = 1
        loss_ex = 100
        count = 0
    
        while delta_loss > 1e-3:
            z_recall.retain_grad()
            optimizer.zero_grad()
            estimated_flow = Trans_model.decode(z_recall)
            estimated_flow = estimated_flow * max_norm
            tf_x = transform(x_c, estimated_flow, mode='bilinear')
            loss = flow_criterion(tf_x * mask, x_predc * mask)/n_seq
            loss.backward(retain_graph=True)
            optimizer.step()

            delta_loss = np.abs(loss_ex - loss.item())
            loss_ex = loss.item()

            count += 1

test_data_path = './data/cardiac_seq'
test_set = Dataset_seq(test_data_path)

# loading the data
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=False)


start = time.time()
test()
end = time.time()
print("training took {:.8f}".format(end-start))