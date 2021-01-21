import torch
from torch import nn

# Reminder Conv2d/ConvTranspose2d args: (in, out, kern, stride, padding)
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.network = nn.Sequential(
      nn.ConvTranspose2d(60, 512, 15, 2, 0, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(True),

      nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True),

      nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),

      nn.ConvTranspose2d(128, 4, 4, 2, 1, bias=False),
      nn.Tanh()
    )

  def forward(self, x):
    return self.network(x)

# Reminder Conv2d/ConvTranspose2d args: (in, out, kern, stride, padding)
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.network = nn.Sequential(
      nn.Conv2d(4, 128, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(256, 512, 4, 2, 1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(512, 1, 15, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.network(x)

def init(model):
  def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0)

  model.apply(weights_init)

import os
from datetime import datetime
from google.cloud import storage
client = storage.Client(project='aesthetic-fx-300721')

def get_model(load=False):
  if load:

    # on kaggle models are saved to FINALX.pt
    # (where X is G or D)
    bucket = client.bucket('python-trained-models')
    blob = bucket.blob('FINALG.pt')
    blob.download_to_filename('FINALG.pt')
    blob = bucket.blob('FINALD.pt')
    blob.download_to_filename('FINALD.pt')

    G = Generator()
    G.load_state_dict(torch.load('FINALG.pt'))
    D = Discriminator()
    D.load_state_dict(torch.load('FINALD.pt'))
    return G, D

  else:
    G = Generator()
    init(G)
    D = Discriminator()
    init(D)
    return G, D

def save_model(model, prefix, suffix):
  filename = f'{prefix}{suffix}.pt'
  torch.save(model.state_dict(), filename)
  bucket = client.bucket('python-trained-models')
  blob = bucket.blob(f'PokeGAN/{prefix}{suffix}.pt')
  blob.upload_from_filename(f'{prefix}{suffix}.pt')
