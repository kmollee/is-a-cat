import os
from fastai.vision.all import *
from fastbook import *
from fastai.vision.widgets import *
import fastbook

fastbook.setup_book()


is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')


path = untar_data(URLs.PETS)/'images'


def is_cat(x): return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

if is_kaggle:
    learn.export("/kaggle/working/model.pkl")
else:
    learn.export("./model.pkl")
