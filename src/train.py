import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from model import save_model

def train(G, D, dataset, epochs, batch_size, lr, betas, weight_decay):

  if torch.cuda.is_available(): device = f'cuda:{torch.cuda.current_device()}'
  else: device = 'cpu'

  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  criterion = nn.BCELoss()
  optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
  optimizerG = optim.Adam(G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
  G.to(device)
  D.to(device)

  print(f'-- training on {device} --')
  for epoch in range(epochs):
    cumulative = 0
    for imgs in loader:

      # sz of batch in case not all batches are same sz
      sz = len(imgs)
    
      # noise on labels
      fuzz = 0.04 * (torch.rand(sz, device=device) - 0.5)

      # pass the real data through D and back prop
      # labels: binary classification
      # 1 -> real
      # 0 -> fake
      D.zero_grad()
      imgs = imgs.to(device)
      real_labels = torch.full((sz, ), 1, dtype=torch.float, device=device)
      real_preds = D(imgs).view(-1)
      real_loss = criterion(real_preds, real_labels)
      real_loss.backward()

      # pass the fake data through D and back prop
      # the size of the latent vector is 60!
      noise = torch.randn(sz, 60, 1, 1, device=device)
      fake_labels = torch.full((sz, ), 0, dtype=torch.float, device=device)
      fake_imgs = G(noise)

      # detach is important to not back propagate to generator here
      fake_preds = D(fake_imgs.detach()).view(-1)
      fake_loss = criterion(fake_preds, fake_labels)
      fake_loss.backward()
      optimizerD.step()

      # train the generator
      # real labels are used because we are rewarding G for "fooling" D
      G.zero_grad()
      generator_loss = criterion(D(fake_imgs).view(-1), real_labels + fuzz)
      cumulative += generator_loss.item()
      generator_loss.backward()
      optimizerG.step()
        
    if epoch % 100 == 0:
      print(f'fake loss: {fake_loss.item():.2f}, real loss: {real_loss.item():.2f}, generator loss: {generator_loss.item():.2f}')
      print(f'epoch {epoch}, loss: {cumulative:.2f}')
      print()

    if epoch % 300 == 0:
      save_model(G, f'{epoch}epochs', 'G')
      save_model(D, f'{epoch}epochs', 'D')
    