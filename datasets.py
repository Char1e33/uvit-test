from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token, targets=None):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token
        self._targets = targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y

    @property
    def targets(self):
        return self._targets

class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10

class CIFAR10(DatasetFactory):
    r""" CIFAR10 dataset

    Information of the raw dataset:
         train: 50,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, path, random_flip=False, cfg=False, p_uncond=None):
        super().__init__()

        transform_train = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        transform_test = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        self.train = datasets.CIFAR10(path, train=True, transform=transform_train, download=True)
        self.test = datasets.CIFAR10(path, train=False, transform=transform_test, download=True)

        assert len(self.train.targets) == 50000
        self.K = max(self.train.targets) + 1
        self.cnt = torch.tensor([len(np.where(np.array(self.train.targets) == k)[0]) for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt: {self.cnt}')
        print(f'frac: {self.frac}')

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 3, 32, 32

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_cifar10_train_pytorch.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# ImageNet


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        if "imagenet" in self.path:
            return 1_281_167 * 2  # consider the random flip
        elif "ChestXray14" in self.path:
            return 64352 * 2
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        data = np.load(path, allow_pickle=True).item()
        z = data['moment']
        label = data['lb']

        return z, label
    
    @property
    def targets(self):
        return self.labels


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * width * width
    """

    def __init__(self, path, resolution=64):
        super().__init__()

        self.resolution = resolution

        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose([Crop(x1, x2, y1, y2), transforms.Resize(self.resolution),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = datasets.CelebA(root=path, split="train", target_type=[], transform=transform, download=True)
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz'

    @property
    def has_label(self):
        return False


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target



def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'

class ChestXray14Database(Dataset):
    def __init__(self, root="/storage/dataset/ChestXray14", csv_file="/storage/dataset/ChestXray14/reports/train_11_1.csv", size=1024, mode='train'):
        """
        Args:
            root (str): 数据集根目录，包含图像和报告的子目录。
            csv_file (str): 包含图像和标签对应关系的CSV文件路径。
            size (int, optional): 图像的目标尺寸（高度和宽度）。默认为None。
        Function:
            用于提取文本和图像特征。
        """
        self.root = root
        self.height = self.width = size
        self.df = pd.read_csv(csv_file)
        self.keys = self.df['Image Index'].tolist()
        self.mode = mode

    def _load_image(self, key: str):
        path = os.path.join(self.root, 'images', key)
        return Image.open(path).convert("RGB")

    def _load_target(self, key: str) -> dict:
        # 获取标签
        labels = self.df[self.df['Image Index'] == key].iloc[:, 2:].values.flatten().tolist()
        label = labels.index(1)
        
        # 获取报告
        report_path = os.path.join(self.root, 'reports/'+self.mode, key.replace('.png', '_0.txt'))
        with open(report_path, 'r') as f:
            report = f.read()

        return {'label': label, 'report': report}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        
        if self.width and self.height:
            image = center_crop(self.width, self.height, image).astype(np.float32)
        
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        target = self._load_target(key)

        return image, target

class ChestXray14FeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        # self.root_query = root + '_query' # 用于SPB的方法，不work
        self.num_data, self.n_captions = get_feature_dir_info(root)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))  # 原始文本数据集
        
        # 从SPB中查询后的文本数据集
        # c = np.load(os.path.join(self.root_query, f'{index}_{k}.npy'))

        return z, c

class ChestXray14_256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset for t2i task...')
        self.train = ChestXray14FeatureDataset(os.path.join(path, 'train'))
        self.test = ChestXray14FeatureDataset(os.path.join(path, 'val'))
        train_path = os.path.join(path, 'train')
        print(f"length of train set {train_path} is ", len(self.train))

        assert len(self.train) == 64352
        assert len(self.test) == 9119
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            data = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            prompt, context = data.item()['prompt'], data.item()['context']
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_ChestXray14.npz'

class ChestXray14_256_ldm_Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset ChestXray14_256_ldm_Features...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 15

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            # labels= []
            # for item in os.listdir(path):
            #     label = np.load(os.path.join(path, item), allow_pickle=True).item()['lb']
            #     labels.append(label)
            #直接读取labels.npy文件，避免读取文件耗费时间
            labels = np.load('/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/assets/ChestXray14_256_ldm_features/labels.npy', allow_pickle=True)
            self.train = CFGDataset(self.train, p_uncond, self.K, targets=labels)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_ChestXray14.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 15, (n_samples,), device=device)


class ISIC2018Database(Dataset):
    def __init__(self, root="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/public/T2I_datasets/ISIC/ISIC2018",
                 csv_file="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/public/T2I_datasets/ISIC/ISIC2018/TrainingGroundTruth.csv", size=1024, mode='train'):
        """
        Args:
            root (str): 数据集根目录，包含图像和报告的子目录。
            csv_file (str): 包含图像和标签对应关系的CSV文件路径。
            size (int, optional): 图像的目标尺寸（高度和宽度）。默认为None。
        Function:
            用于提取文本和图像特征。
        """
        self.root = root
        self.height = self.width = size
        self.df = pd.read_csv(csv_file)
        self.keys = self.df['image'].tolist()
        self.mode = mode

    def _load_image(self, key: str):
        path = os.path.join(self.root, f'{self.mode}_data', key+'.jpg')
        return Image.open(path).convert("RGB")

    def _load_target(self, key: str) -> dict:
        # 获取标签
        # labels = self.df[self.df['image'] == key].iloc[:, 2:].values.flatten().tolist()
        # label = labels.index(1)

        # 获取报告
        report_path = os.path.join(self.root, f'response_910_{self.mode}', key+'.txt')
        with open(report_path, 'r') as f:
            report = f.read()

        return {'report': report}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)

        if self.width and self.height:
            image = center_crop(self.width, self.height, image).astype(np.float32)

        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        target = self._load_target(key)

        return image, target


class ISIC2018FeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        # self.root_query = root + '_query' # 用于SPB的方法，不work
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))  # 原始文本数据集

        # 从SPB中查询后的文本数据集
        # c = np.load(os.path.join(self.root_query, f'{index}_{k}.npy'))

        return z, c


class ISIC2018_256Features(
    DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset for t2i task...')
        self.train = ISIC2018FeatureDataset(os.path.join(path, 'train'))
        self.test = ISIC2018FeatureDataset(os.path.join(path, 'test'))
        train_path = os.path.join(path, 'train')
        print(f"length of train set {train_path} is ", len(self.train))

        assert len(self.train) == 10015
        assert len(self.test) == 1512
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            data = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            prompt, context = data.item()['prompt'], data.item()['context']
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/public/T2I_datasets/fid_states/ISIC2018_train.npz'



def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == 'ChestXray14_features':
        return ChestXray14_256Features(**kwargs)
    elif name == 'ChestXray14_256_ldm_features':
        return ChestXray14_256_ldm_Features(**kwargs)
    elif name == 'ISIC2018_Features':
        return ISIC2018_256Features(**kwargs)
    else:
        raise NotImplementedError(name)

