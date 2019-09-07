from fastai.vision import *

folder = 'person'
file = 'urls_person.csv'

folder = 'anime'
file = 'urls_anime.csv'

classes = ['person', 'anime']

path = './data'
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)

learn.fit_one_cycle(2, max_lr=slice(1e-4,8e-4))
learn.export('../export.pkl')
