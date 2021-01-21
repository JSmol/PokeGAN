from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Images(Dataset):
  def __init__(self, path, transform=None):
    imgpaths = list(map(str, Path(path).rglob("*.png")))

    self.imgs = list(map(Image.open, imgpaths))
    self.transform = transform

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    if self.transform: return self.transform(self.imgs[idx])
    else: return self.imgs[idx]

from google.cloud import storage
client = storage.Client(project='aesthetic-fx-300721')
bucket = client.bucket('pokemon-images')
def load_data(path):

  path = path.rstrip('/')
  for i in range(1, 802):
    blob = bucket.blob(f'{i:04}.png')
    blob.download_to_filename(f'{path}/{i:04}.png')

  # image transforms
  transform = transforms.Compose([
    transforms.RandomResizedCrop(120, (0.94, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
  ])

  return Images(path, transform=transform)
