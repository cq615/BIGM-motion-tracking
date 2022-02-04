import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
from network import *
from dataio import *
from network import *
from utils import *

lr = 1e-4
n_worker = 4
bs = 1
n_epoch = 200
max_norm = 0.25

model_load_path = './models/main_prior_model_pretrained.pth'
model_save_path = './models/main_prior_model.pth'

n_seq=50
VAE_model = Temporal_VAE(img_size=96, z_dim=32, factor=8, nf=32, n_seq=n_seq)

VAE_model = VAE_model.cuda()
VAE_model.load_state_dict(torch.load(model_load_path))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, VAE_model.parameters()), lr=lr)

Tensor = torch.cuda.FloatTensor

weight = define_weight()
weight = np.tile(weight[np.newaxis], (bs, 1))
weight = torch.tensor(weight, dtype=torch.float, device="cuda:0")
weight = weight.view(-1, )

def train(epoch):
    VAE_model.train()
    epoch_loss = []
    
    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):
        flow, mask = batch

        # reshape sequence for input to encoder
        flow = flow.view(-1, flow.shape[2], flow.shape[3], flow.shape[4]) # simulated deformations
        mask = mask.view(-1, mask.shape[2], mask.shape[3], mask.shape[4]) # dilated myocardial mask

        flow = Variable(flow.type(Tensor))
        mask = Variable(mask.type(Tensor))

        optimizer.zero_grad()
        recon, mu, logvar = VAE_model(flow, mask, max_norm)

        gd_loss = weighted_mse_loss(compute_gradient(flow)*mask.detach(), compute_gradient(recon), weight)
        df_loss = MotionVAELoss_weighted(recon, flow*mask.detach(), weight, mu, logvar,  beta=1e-2)

        loss = df_loss + 10*gd_loss 
        loss.backward()

        optimizer.step()
        epoch_loss.append(loss.item())

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(flow), len(training_data_loader.dataset),
        100. * batch_idx / len(training_data_loader), np.mean(epoch_loss)))


def test():
    VAE_model.eval()
    test_loss = []
    global base_err

    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        flow, mask = batch

        flow = flow.view(-1, flow.shape[2], flow.shape[3], flow.shape[4])
        mask = mask.view(-1, mask.shape[2], mask.shape[3], mask.shape[4])

        flow = Variable(flow.type(Tensor))
        mask = Variable(mask.type(Tensor))

        recon, mu, logvar = VAE_model(flow, mask, max_norm)

        gd_loss = weighted_mse_loss(compute_gradient(flow)*mask.detach(), compute_gradient(recon), weight)
        df_loss = MotionVAELoss_weighted(recon, flow*mask.detach(), weight, mu, logvar,  beta=1e-2)

        loss = df_loss + 10 * gd_loss 
        test_loss.append(loss.item())
  
    print('Base Loss: {:.6f}'.format(base_err))
    print('Test Loss: {:.6f}'.format(np.mean(test_loss)))

    if np.mean(test_loss) < base_err:
        torch.save(VAE_model.state_dict(), model_save_path)
        print("Checkpoint saved to {}".format(model_save_path))
        base_err = np.mean(test_loss)


data_path = './data/4DSimMotion'
train_set = Dataset_motion(data_path, 'train')
test_set = Dataset_motion(data_path, 'val')

# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=False)
base_err = 1000

for epoch in range(0, n_epoch + 1):

    print('Epoch {}'.format(epoch))
    start = time.time()
    test()
    end = time.time()
    print("testing took {:.8f}".format(end-start))

    start = time.time()
    train(epoch)
    end = time.time()
    print("training took {:.8f}".format(end-start))