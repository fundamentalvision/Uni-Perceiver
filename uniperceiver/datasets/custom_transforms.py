from torchvision import transforms as T
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


def clip_transforms(mode='train', img_size=224, flip_prob=0.5):
    assert mode in ['train', 'test', 'val']
    min_size = img_size
    max_size = img_size
    # assert min_size <= max_size


    normalize_transform = T.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    if mode == 'train':
        transform = T.Compose(
            [
                T.Resize(max_size, BICUBIC),
                T.RandomCrop(min_size),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(max_size, BICUBIC),
                T.CenterCrop(min_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
