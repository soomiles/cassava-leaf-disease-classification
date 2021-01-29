from albumentations import *
from albumentations.pytorch import ToTensorV2


mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def get_transforms(need=('train', 'val'), img_size=(384, 384), crop=True):
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            RandomResizedCrop(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Transpose(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'val' in need:
        new_size = tuple([int((576 / 512) * size) for size in img_size]) if crop else img_size
        transformations['val'] = Compose([
            PadIfNeeded(min_height=512, min_width=512),
            CenterCrop(img_size[0], img_size[1]),
            Resize(new_size[0], new_size[1]),
            CenterCrop(img_size[0], img_size[1]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations
