import os
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, imglist, transform=None):
        self.image_paths = imglist
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

    
def get_loader(image_path, image_size, batch_size, num_workers=2):
    """Builds and returns Dataloader."""
    
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = ImageFolder(image_path, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return dataset