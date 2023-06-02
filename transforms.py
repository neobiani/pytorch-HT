

import torchio as tio

def get_transforms():
    transforms_train=[
        #tio.RescaleIntensity(out_min_max=(0,1)),
        tio.ZNormalization(),
        tio.RandomNoise(),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=3),
        tio.RandomFlip(axes=(1,)),]

    transforms_valid=[
        #tio.RescaleIntensity(out_min_max=(0,1)),
        tio.ZNormalization(),]

    transform_train=tio.Compose(transforms_train)
    transform_valid=tio.Compose(transforms_valid)
    return transform_train, transform_valid

