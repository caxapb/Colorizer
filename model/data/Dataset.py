from torch.utils.data import Dataset
from data.functions import get_image, check_url
import polars as pl
from data.transforms import transform
from torch import Tensor, max
from tqdm import tqdm
from torchvision import transforms


class PictureDataset(Dataset):
    def __init__(self, urls):
        checked = []
        for url in tqdm(urls["url"]):
            checked.append(check_url(url))

        urls = urls.filter(checked)
        self.urls = urls

        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((2048, 2048)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):

        image = get_image(self.urls["url"][idx])

        gray = transform(image)
        print(image.shape, gray.shape)
        image, gray = self.resize_transform(image), self.resize_transform(gray)
        print(image.shape, gray.shape)

        # gray = gray.transpose(0, 2)
        # gray = gray.transpose(1, 2)
        # image = image.transpose(0, 2)
        # image = image.transpose(1, 2)
        image = image/max(image)

        gray = gray/max(gray)

        return image, gray