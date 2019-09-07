from fastai.vision import *
import os

learn = load_learner('./', 'final.pkl')

path = Path('./eval')
for filename in os.listdir(path):
    img = open_image(path/filename)
    pred_class,pred_idx,outputs = learn.predict(img)
    message = str(outputs[0].item()) + ' anime ' + str(outputs[1].item()) + ' person'
    print(filename, message)
