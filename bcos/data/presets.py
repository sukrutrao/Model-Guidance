import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

import bcos.data.transforms as custom_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        is_bcos=False,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(
                        interpolation=interpolation, magnitude=ra_magnitude
                    )
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(
                        interpolation=interpolation, severity=augmix_severity
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if not is_bcos:
            trans.append(transforms.Normalize(mean=mean, std=std))
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        if is_bcos:
            trans.append(custom_transforms.AddInverse())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        # https://rich.readthedocs.io/en/stable/pretty.html
        yield "transforms", self.transforms.transforms


class ImageNetClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        is_bcos=False,
    ):
        trans = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


class CIFAR10ClassificationPresetTrain:
    def __init__(
        self,
        *,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


class CIFAR10ClassificationPresetTest:
    def __init__(
        self,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms
