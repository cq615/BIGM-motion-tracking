import torch
from torch import nn
import time
import torch.nn.functional as F

# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# UnFlatten layer
class UnFlatten(nn.Module):
    def __init__(self, C, H, W):
        super(UnFlatten, self).__init__()
        self.C, self.H, self.W = C, H, W

    def forward(self, input):
        return input.view(input.size(0), self.C, self.H, self.W)

class Temporal_VAE(nn.Module):
    def __init__(self, img_size=96, z_dim=32, factor=8, nf=32, n_seq=25):
        super(Temporal_VAE, self).__init__()
        self.n_seq = n_seq
        self.z_dim = z_dim
        self.factor = factor
        # input 1 x n x n
        self.conv1 = nn.Conv2d(2, nf, kernel_size=4, stride=2, padding=1)
        # size nf x n/2 x n/2
        self.conv2 = nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x n/4 x n/4
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x n/8 x n/8
        self.conv4 = nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x n/16*n/16

        h_dim = int(nf * 8 * img_size / 16 * img_size / 16)
        self.h_dim = h_dim

        self.fc11 = nn.RNN(h_dim, z_dim, nonlinearity='tanh', batch_first=True)
        self.fc12 = nn.RNN(h_dim, z_dim, nonlinearity='tanh', batch_first=True)

        self.rnn = nn.RNN(z_dim, factor*z_dim, nonlinearity='tanh', batch_first=True)
        self.fc2 = nn.Linear(factor*z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv22 = nn.Conv2d(nf*2, nf*2, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv32 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1)
        self.deconv42 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        self.deconv21 = nn.Conv2d(nf*2, 2, kernel_size=3, padding=1)
        self.deconv31 = nn.Conv2d(nf, 2, kernel_size=3, padding=1)
        self.deconv41 = nn.Conv2d(nf, 2, kernel_size=3, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            self.conv4,
            nn.ReLU(),
            Flatten()
        )

        self.unflatten = UnFlatten(C=int(nf * 8), H=int(img_size / 16), W=int(img_size / 16))

    def decoder(self, z):
        z, _ = self.rnn(z.view(-1, self.n_seq, self.z_dim))
        z = z.contiguous().view(-1, self.factor*self.z_dim)
        z = self.fc2(z)
        z = self.unflatten(z)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv22(z))
        out1 = F.tanh(self.deconv21(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv32(z))
        out2 = F.tanh(self.deconv31(z))
        out2 = out2 + F.interpolate(out1, scale_factor=2, mode='bilinear')
        z = F.relu(self.deconv4(z))
        z = F.relu(self.deconv42(z))
        out3 = F.tanh(self.deconv41(z))
        out3 = out3 + F.interpolate(out2, scale_factor=2, mode='bilinear')
        return out3

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        h = h.view(-1, self.n_seq, self.h_dim)
        mu, _= self.fc11(h)
        logvar, _ = self.fc12(h)
        mu = mu.contiguous().view(-1, self.z_dim)
        logvar = logvar.contiguous().view(-1, self.z_dim)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):  
        out = self.decoder(z)
        return out

    def forward(self, x, mask, max_norm):
        x = x * mask
        x = x/max_norm
        z, mu, logvar = self.encode(x)
        out = self.decode(z)
        return out*max_norm, mu, logvar