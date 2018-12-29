import argparse
# get help by `python car_model.py -h`
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of vehicles.csv)")
ap.add_argument("-m", "--model", required=True,
    help="path to Keras dual-branched DNN model")
ap.add_argument("-o", "--output", required=True,
    help="path to output csv file")

args = vars(ap.parse_args())

import keras
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import pandas as pd 
import numpy as np


# parsed arguments
fpath = args["dataset"] # feature data folder
mpath = args["model"] # model folder
output = args["output"] # output file
# import data
x = pd.read_csv(fpath+'/vehicles.csv')
x = x.iloc[:,1:].values

# normalize the data
x = (x-x.mean())/x.std()
model = keras.models.load_model(mpath+'/best_model.h5')
y_pred = model.predict_on_batch(x) # predict on a single batch

lb = LabelBinarizer()
lb.fit_transform([0,1,2])
d = {'n_reservation':[i[0] for i in y_pred[0]], 'reservation_type':list(lb.inverse_transform(y_pred[1]))}
df = pd.DataFrame(data=d)
df.to_csv(output)
print('total reservation predicted = %s' %str(sum(y_pred[0])))
print('finished saving output to file: %s' % output)
