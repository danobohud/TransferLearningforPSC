import tensorflow as tf
from Helpers import writeline

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

sess = InteractiveSession(config=config)

# Import libraries

import numpy as np
import argparse, datetime, csv, os, h5py, pickle
import matplotlib

matplotlib.use('Agg')
from keras.applications import DenseNet121
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from sklearn.metrics import classification_report,accuracy_score
from keras.optimizers import Adam

def load_traintest(filepath):
    '''Load train and test files from .h5'''

    print('Loading data from %s:'%(filepath))
    if filepath[-3:] != '.h5':
        filepath += '.h5'
    h5f = h5py.File(filepath, 'r')
    X_train, C_train,A_train,T_train,H_train, _,X_test,C_test,A_test,T_test,H_test,_,nums = [np.asarray(h5f[x]) for x in [
        'X_train', 'C_train', 'A_train', 'T_train', 'H_train',
        'Train_domains', 'X_test', 'C_test', 'A_test', 'T_test',
        'H_test', 'Test_domains', 'Classes']]
    return (X_train, (C_train, A_train, T_train, H_train), X_test, (C_test, A_test, T_test, H_test)), nums

def extract_labels(input_labels):
    '''Convert labels from input into into separate arrays'''

    C = np.asarray([i for i in input_labels[0]])
    A = np.asarray([i for i in input_labels[1]])
    T = np.asarray([i for i in input_labels[2]])
    H = np.asarray([i for i in input_labels[3]])

    return C, A, T, H

def getmax(prediction):
    '''Return the maximum from a onehot array'''

    converted = []
    for l in range(len(prediction)):
        idx = np.argmax(prediction[l], axis=-1)
        out = np.zeros(prediction[l].shape)
        out[np.arange(out.shape[0]), idx] = 1
        converted.append(out)
    return converted


class BaseLine(object):
    '''Create adapted DenseNet121 model'''

    def __init__(self, inputs, savedir, input_file, batch_size=False):
        self.savedir = savedir
        self.input_file = input_file.split('/')[-1][:-3]
        self.preprocessed_data = inputs[0]
        self.nums = inputs[1]
        self.scaleDim = 255

        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.epochs = 0
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = 32

    def build(self, output_layer=False):
        '''Model object: Initialise, build, train and/or evaluate model. Adapted from
        https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/ '''

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        (numC, numA, numT, numH) = self.nums

        # Select pre-trained model from keras
        with strategy.scope():
            base_model = DenseNet121(weights='imagenet', include_top=True,
                                     input_tensor=Input(shape=(self.scaleDim, self.scaleDim, 3)))

            # Remove the last layer
            base_model.layers.pop()
            print('\nBase model: {}\nNumber of Layers: {}'.format(base_model.name, len(base_model.layers)))
            self.name = base_model.name + '_ENSEMBLE'

            # Add output layers and softmax heads

            if output_layer:
                self.output_layer = output_layer
            else:
                self.output_layer = 512

            x = base_model.output
            PFP = Dense(self.output_layer, input_shape=(2048,), activation='relu', name='PFP')(x)
            BN = BatchNormalization(name='BatchNorm')(PFP)
            Drop = Dropout(0.25)(BN)
            C_labels = Dense(numC, input_shape=(self.output_layer,), activation='softmax', name='C_labels')(Drop)
            A_labels = Dense(numA, input_shape=(self.output_layer,), activation='softmax', name='A_labels')(Drop)
            T_labels = Dense(numT, input_shape=(self.output_layer,), activation='softmax', name='T_labels')(Drop)
            H_labels = Dense(numH, input_shape=(self.output_layer,), activation='softmax', name='H_labels')(Drop)

            # Compile


            self.model = Model(inputs=base_model.input, outputs=[C_labels, A_labels, T_labels, H_labels])

            opt = Adam()
            self.model.compile(optimizer=opt,
                               loss={"C_labels": 'categorical_crossentropy', "A_labels": 'categorical_crossentropy',
                                     "T_labels": 'categorical_crossentropy', "H_labels": 'categorical_crossentropy'},
                               metrics=['accuracy'])

def getpreds(prediction, weight):
    '''Apply weighting to softmax probabilities'''
    print('Weighting')
    ensemble_preds = []
    for l in range(len(prediction)):
        probs = [[], []]
        for i in range(len(prediction[l])):         # For each task
            idx = np.argmax(prediction[l][i], axis=-1)      # Get the index (class) of the most likely prediction
            p = prediction[l][i][idx]                       # Get the probability of that prediction
            probs[0].append(idx)                            # Record the index
            probs[1].append(p * weight)                     # Record the weighted maximum probability
        ensemble_preds.append(probs)

    return ensemble_preds


def weighted_prediction(ensemble_preds, ypred):
    '''Get weighted ensenble predictions for C,A, T and H tasks'''
    out = []
    for t in range(len(ypred)):
        output = np.zeros(np.shape(ypred[t]))       # Create output matrix for each task
        probs = [[m[t][1][i] for m in ensemble_preds] for i in range(len(ypred[t]))]    # Get the maximum probabilities from each model for each instance
        ids = [[x[t][0][i] for x in ensemble_preds] for i in range(len(ypred[t]))]      # Get the indices for each of those probabilities
        best = [x.index(np.max(x)) for x in probs]                                      # Find the most certain weighted prediction over all models for all instances
        idx = [ids[id][best[id]] for id in range(len(ids))]                             # Find the indices (predicted class) for those predictions
        output[np.arange(output.shape[0]), idx] = 1                                     # Set the predicted class as that class
        out.append(output)
    return out


def ensemble_prediction(baseline, loadfiles):
    '''Create four pre-trained DenseNet121 models, and return the maximum weighted prediction
    over all models for each instance'''

    p = []

    weights = [1 / len(loadfiles)] * len(loadfiles)

    print('Evaluating accuracy on test data')
    _, _, X, _ = baseline.preprocessed_data

    for i, loadfile in enumerate(loadfiles):
        print('\nModel :', str(i + 1))
        print('Loading weights from ', loadfile)
        baseline.model.load_weights(loadfile)
        print('Getting predictions')
        ypred = baseline.model.predict(X)   # Make predictions
        maxpreds = getpreds(ypred, weights[i])      # Weight the predictions
        p.append(maxpreds)                        # Return the maximum weighted probability per instance
    ypred_new = weighted_prediction(p, ypred)
    return ypred_new


def get_report(baseline, writer, report, testresults):
    names = ['C', 'A', 'T', 'H']
    for i in range(0, len(report)):
        writer.writerow([baseline.time, baseline.name, baseline.output_layer, baseline.epochs, baseline.batch_size,
                         names[i], 'accuracy', testresults[i], '-', '-', '-', '-', baseline.input_file])
        for item in report[i].keys():
            writer.writerow([baseline.time, baseline.name, baseline.output_layer, baseline.epochs, baseline.batch_size,
                             names[i], item, '-', report[i][item]['precision'],
                             report[i][item]['recall'],
                             report[i][item]['f1-score'],
                             report[i][item]['support'], baseline.input_file])


def save_results(baseline, ypred_new,save_f1=False):
    print('\nResults for weighted ensemble: \n')
    _, _, X, y = baseline.preprocessed_data
    y = [i for i in extract_labels(y)]
    reports = [classification_report(y[i], ypred_new[i], output_dict=True, digits=4) for i in np.arange(0, 4)]
    names = ['Class', 'Architecture', 'Topology', 'Homologous Superfamily']
    accs = [accuracy_score(y[i], ypred_new[i]) for i in range(len(y))]
    for i in range(4):
        print('{} - Acc: {:.2f}%, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Support: {},'.format(names[i], accs[i],
                                                                                            reports[i]['weighted avg'][
                                                                                                'precision'] * 100,
                                                                                            reports[i]['weighted avg'][
                                                                                                'recall'] * 100,
                                                                                            reports[i]['weighted avg'][
                                                                                                'f1-score'] * 100,
                                                                                            reports[i]['weighted avg'][
                                                                                                'support']))
    if save_f1:
        f1_file = os.path.join(baseline.savedir,'Results/F1.csv')
        print('Saving F1 to ', f1_file)
        with open(f1_file, 'a') as f:
            writer = csv.writer(f)
            get_report(baseline, writer, reports, accs)



def run_model(input_file, save_f1=False,
              batch_size=False, select_ensemble=False):

    if not select_ensemble:
        ensemble='e2'
    else:
        ensemble=select_ensemble
    savedir=os.getcwd()
    inputs = load_traintest(input_file)
    print('Intialising')
    baseline = BaseLine(inputs, savedir, input_file, batch_size)  # Initialise model
    print('Building model')
    baseline.build()
    if ensemble=='e2':
        loadfiles = [os.path.join(savedir, x) for x in
                     ['HRBB_BEST_150.ckpt', 'HRBB_NORM_BEST_150.ckpt', 'HRCA_BEST_150.ckpt', 'HRHEAVY_BEST_150.ckpt']]
    elif ensemble =='e1':
        loadfiles = [os.path.join(savedir, x) for x in
                     ['HRBB_BEST_150.ckpt', 'HRCA_BEST_150.ckpt', 'HRHEAVY_BEST_150.ckpt']]
    else:
        print('Error, please select an ensemble from "e1" or "e2"')

    ypred_new = ensemble_prediction(baseline, loadfiles)
    save_results(baseline, ypred_new,save_f1)
    print('\nComplete')

if __name__ == '__main__':
    # Handle command line options
    parser = argparse.ArgumentParser(description='Run Baseline Supervised Learning Model')
    parser.add_argument('-i', '--input_file', required=True,
                        help='Target hf5 file')
    parser.add_argument('-f1', '--save_f1', required=False, type=str,
                        help='output file for test F1 scores')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        help='Batch size')
    parser.add_argument('-e', '--select_ensemble', required=False, type=str,
                        help='Select E1 or E2')
    args = parser.parse_args()


    if args.select_ensemble not in ['e1','e2']:
        print('Error, please select one of "e1","e2"')
    run_model(args.input_file, args.save_f1, args.batch_size,args.select_ensemble)
