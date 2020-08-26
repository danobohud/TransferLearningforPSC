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
import matplotlib.pyplot as plt
from time import time
from keras.applications import DenseNet121
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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
    '''Model object: Initialise, build, train and/or evaluate model. Adapted from
        https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/ and
        https://www.tensorflow.org/guide/keras/train_and_evaluate'''

    def __init__(self, inputs, savedir, input_file, epochs, batch_size=False):

        '''Set model parameters from input specifications'''
        self.savedir = savedir
        self.input_file = input_file.split('/')[-1][:-3]
        self.preprocessed_data = inputs[0]
        self.nums = inputs[1]
        self.scaleDim = 255
        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = 32

    def build(self, output_layer=False, learning_rate=False, loadfile=False):
        '''Build multi-task DenseNet121 model.'''

        strategy = tf.distribute.MirroredStrategy()                                         # Set distributed strategy
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        (numC, numA, numT, numH) = self.nums                                                # Get number of classes
        with strategy.scope():
            base_model = DenseNet121(weights='imagenet', include_top=True,                      # Import DenseNet121
                                     input_tensor=Input(shape=(self.scaleDim, self.scaleDim, 3)))

            base_model.layers.pop()                                                             # Remove the last layer
            print('Base model: {}\nNumber of Layers: {}'.format(base_model.name,
                                                                    len(base_model.layers)))
            self.name = base_model.name

            if output_layer:
                self.output_layer = output_layer        # Set width of PFP layer
            else:
                self.output_layer = 512

            # Amend DenseNet backbone for multi-task learning

            x = base_model.output
            PFP = Dense(self.output_layer, input_shape=(2048,), activation='relu', name='PFP')(x)   # Protein fingerprint layer
            BN = BatchNormalization(name='BatchNorm')(PFP)
            Drop = Dropout(0.25)(BN)
            C_labels = Dense(numC, input_shape=(self.output_layer,), activation='softmax', name='C_labels')(Drop)
            A_labels = Dense(numA, input_shape=(self.output_layer,), activation='softmax', name='A_labels')(Drop)
            T_labels = Dense(numT, input_shape=(self.output_layer,), activation='softmax', name='T_labels')(Drop)
            H_labels = Dense(numH, input_shape=(self.output_layer,), activation='softmax', name='H_labels')(Drop)

            # Compile

            self.model = Model(inputs=base_model.input, outputs=[C_labels, A_labels, T_labels, H_labels])

            if not learning_rate:
                lr = 0.001
            else:
                lr = learning_rate

            if loadfile:
            # Load weights from pre-trained model
                print('Loading model weights from ', loadfile)
                self.model.load_weights(loadfile)
                print('Loaded')

            lossWeights = {"C_labels": 1.0, "A_labels": 1.0, "T_labels": 1.0, "H_labels": 5.0}
            opt = Adam(learning_rate=lr)
            self.model.compile(optimizer=opt,
                               loss={"C_labels": 'categorical_crossentropy', "A_labels": 'categorical_crossentropy',
                                     "T_labels": 'categorical_crossentropy', "H_labels": 'categorical_crossentropy'},
                               loss_weights=lossWeights,
                               metrics=['accuracy'])

    def train(self, save=False):
        '''Set training configuration'''

        X_train, train_labels, _, _ = self.preprocessed_data
        C_train, A_train, T_train, H_train = train_labels

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.00001)
        start = time()

        if save:            # Save model checkpoints
            checkpoint = self.savedir + '/Models/' + str(self.time) + '_' + self.name + '_' + self.input_file
            os.mkdir(checkpoint)
            checkpoint_path = checkpoint + "/{epoch:04d}.ckpt"
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          verbose=1, period=10)

            self.history = self.model.fit(X_train, y={"C_labels": C_train, "A_labels": A_train, "T_labels": T_train,
                                                      "H_labels": H_train},
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_split=0.4,
                                          verbose=1, callbacks=[cp_callback, reduce_lr])

        else:
            self.history = self.model.fit(X_train, y={"C_labels": C_train, "A_labels": A_train, "T_labels": T_train,
                                                      "H_labels": H_train},
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_split=0.4,
                                          verbose=1, callbacks=[reduce_lr])

        self.train_time = time() - start

    def plot(self):
        '''Plot loss and accuracy '''

        self.plotdir = self.savedir + '/Plots/'
        lossNames = ["loss", "C_labels_loss", "A_labels_loss", "T_labels_loss", "H_labels_loss"]
        EPOCHS = len(self.history.history['val_loss'])
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots(5, 1, figsize=(13, 13))
        for (i, l) in enumerate(lossNames):
            title = "Loss for {}".format(l) if l != "loss" else "Total loss"
            ax[i].set_title(title)
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel("Loss")
            ax[i].plot(np.arange(0, EPOCHS), self.history.history[l], label=l)
            ax[i].plot(np.arange(0, EPOCHS), self.history.history["val_" + l],
                       label="val_" + l)
            ax[i].legend()
        plt.tight_layout()
        plt.savefig(self.plotdir + self.time + str(self.name) + '_' + str(self.output_layer) + "{}_losses.png".format(
            '_train_ca'))
        plt.close()

        accuracyNames = ["C_labels_accuracy", "A_labels_accuracy", "T_labels_accuracy", "H_labels_accuracy"]
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots(len(accuracyNames), 1, figsize=(8, 8))
        for (i, l) in enumerate(accuracyNames):
            ax[i].set_title("Accuracy for {}".format(l))
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel("Accuracy")
            ax[i].plot(np.arange(0, EPOCHS), self.history.history[l], label=l)
            ax[i].plot(np.arange(0, EPOCHS), self.history.history["val_" + l],
                       label="val_" + l)
            ax[i].legend()
        plt.tight_layout()
        plt.savefig(
            self.plotdir + self.time + str(self.name) + '_' + str(self.output_layer) + "{}_acc.png".format('_train_ca'))
        plt.close()

    def get_accuracy(self, X, y, train=False):
        '''Save accuracy to results file'''

        results = []
        if train:
            print('Saving training and validation accuracy')
            lossNames = ["loss", "C_labels_accuracy", "A_labels_accuracy", "T_labels_accuracy", "H_labels_accuracy"]
            lossNames.extend(['val_' + item for item in lossNames])
            for item in lossNames:
                results.append(self.history.history[item][-1])

        self.testresults = self.model.evaluate(X, y)
        print('\nAccuracy:\n\nC: {:.2f}%\nA: {:.2f}%\nT: {:.2f}%\nH: {:.2f}%'.format(self.testresults[-4] * 100,
                                                                                     self.testresults[-3] * 100,
                                                                                     self.testresults[-2] * 100,
                                                                                     self.testresults[-1] * 100, ))
        results.append(self.testresults[0])
        results.extend(self.testresults[-4:])

        return results

    def save_accuracy(self, resultsfile, train=False):
        '''Save test time accuracy'''

        print('\nEvaluating accuracy on test data')
        _, _, X, y = self.preprocessed_data
        y = [i for i in extract_labels(y)]
        results = self.get_accuracy(X, y, train)
        if not train:
            self.train_time = 0
            insert = [self.time, self.name, self.output_layer, self.epochs, self.batch_size,'-','-','-','-','-','-',
                      '-','-','-','-',]
        else:
            insert = [self.time, self.name, self.output_layer, self.epochs, self.batch_size]
        results = insert + results + [self.train_time, self.input_file]
        print('Saving accuracy to ', resultsfile)
        writeline(resultsfile,results)


    def get_report(self, writer, report, testresults):
        '''Compute F1 scores for each task and category'''
        names = ['C', 'A', 'T', 'H']
        acc_idx = [-4, -3, -2, -1]
        for i in range(0, len(report)):
            writer.writerow([self.time, self.name, self.output_layer, self.epochs, self.batch_size,
                             names[i], 'accuracy', testresults[acc_idx[i]], '-', '-', '-', '-', self.input_file])
            for item in report[i].keys():
                writer.writerow([self.time, self.name, self.output_layer, self.epochs, self.batch_size,
                                 names[i], item, '-', report[i][item]['precision'],
                                 report[i][item]['recall'],
                                 report[i][item]['f1-score'],
                                 report[i][item]['support'], self.input_file])

    def save_report(self, f1):
        '''Save F1 scores'''
        print('\nEvaluating F1 on test data')
        _, _, X, y = self.preprocessed_data
        y = [i for i in extract_labels(y)]
        print('Getting classification report (weighted averages):')
        self.ypred = getmax(self.model.predict(X))
        reports = [classification_report(y[i], self.ypred[i], output_dict=True, digits=4) for i in np.arange(0, 4)]
        names = ['Class', 'Architecture', 'Topology', 'Homologous Superfamily']
        for i in range(4):
            print('{} - P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Support: {},'.format(names[i],
                                                                                  reports[i]['weighted avg'][
                                                                                      'precision'] * 100,
                                                                                  reports[i]['weighted avg'][
                                                                                      'recall'] * 100,
                                                                                  reports[i]['weighted avg'][
                                                                                      'f1-score'] * 100,
                                                                                  reports[i]['weighted avg'][
                                                                                      'support']))
        print('Saving results to ', f1)
        with open(f1, 'a') as f:
            writer = csv.writer(f)
            self.get_report(writer, reports, self.testresults)

def pfp(trained_model):
    '''Extract learned embeddings from PFP layer. Adapted from
    https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction'''

    if not os.path.isdir(trained_model.savedir + '/Features/'):
        os.mkdir(trained_model.savedir + '/Features/')
    newmodel = Model(inputs=trained_model.model.input, outputs=trained_model.model.layers[-7].output)
    newmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    savefile = trained_model.savedir + '/Features/' + trained_model.time + '_' + trained_model.name + '_' + trained_model.input_file + '_PFP'
    _, _, X, y = trained_model.preprocessed_data
    savefile += '_test_%s' % (len(X))
    y = [i for i in extract_labels(y)]
    print('\nExtracting protein fingerprints')
    features = newmodel.predict(X)
    ypred = getmax(trained_model.model.predict(X))
    savefile = trained_model.savedir + '/Features/' + trained_model.time + '_' + trained_model.name + '_' + trained_model.input_file + '_PFP'
    print('Saving to', savefile)
    np.savez(savefile, features_labels_predicted=(features, y, ypred), allow_pickle=True)
    print('Complete')

def run_model(input_file, savedir, epochs, results=False, output_layer=False,
              loadfile=False, save_f1=False,
              train=False, batch_size=False,
              save_checkpoints=False, freeze=False, lr=False, save_pfp=False):
    """

    :type f1_file: bool
    """
    inputs = load_traintest(input_file)

    print('Intialising')
    baseline = BaseLine(inputs, savedir, input_file, epochs, batch_size)  # Initialise model
    print('Building model')
    baseline.build(output_layer, learning_rate=lr)

    if loadfile:
        print('Loading model weights from ', loadfile)
        baseline.model.load_weights(loadfile)
        print('Loaded')

    if freeze:
        print('Freezing layers')
        for layer in baseline.model.layers[:-8]:
            layer.trainable = False

    if train:
        print('Training model')
        baseline.train(save=save_checkpoints)
        print('Plotting history')
        baseline.plot()
        with open(savedir + '/History/' + baseline.time + '_' + baseline.name + '_' + baseline.input_file + '.pickle',
                  'wb') as p:
            pickle.dump(baseline.model.history.history, p, protocol=2)

    if results:
        resultsfile='{}/Results/results.csv'.format(savedir)
        if not os.path.isfile(resultsfile):
            os.system('touch {}'.format(resultsfile))
            writeline(resultsfile,['Date_time','Model','PFP','Epochs','Batch_size','Loss','C_labels_acc','A_labels_acc','T_labels_acc',
          'H_labels_acc','val_loss','val_C_labels_acc',	'val_A_labels_acc','val_T_labels_acc','val_H_labels_acc',
          'test_loss','test_C_labels_acc','test_A_labels_acc','test_T_labels_acc','test_H_labels_acc',
          'Time','Input_file'])

        baseline.save_accuracy(resultsfile, train)

    if save_f1:
        f1='{}/Results/f1.csv'.format(savedir)
        if not os.path.isfile(f1):
            os.system('touch {}'.format(f1))
            writeline(f1,['ID','Model','PFP','Epochs','Batch size','Task','Class','Accuracy','Precision','Recall','F1-score','Support','Input'])
        baseline.save_report(f1)


    if save_pfp:
        pfp(baseline)

    print('\nModel training and evaluation complete')

if __name__ == '__main__':

    # Handle command line options
    parser = argparse.ArgumentParser(description='Run Baseline Supervised Learning Model')

    parser.add_argument('-i', '--input_file', required=True,
                        help='Target hf5 file')
    parser.add_argument('-e', '--epochs', required=True, type=int,
                        help='Epochs')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        help='Batch size')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float,
                    help='initial learning rate')
    parser.add_argument('-o', '--output_layer', required=False, type=int,
                        help='Width of final dense layer: default 512')
    parser.add_argument('-lf', '--loadfile', required=False, type=str,
                        help='pickle file for model checkpoints')
    parser.add_argument('-r', '--results', required=False, type=bool,
                        help='Save accuracy results')
    parser.add_argument('-f1', '--f1', required=False, type=bool,
                        help='Save F1 results')
    parser.add_argument('-ch', '--save_checkpoints', required=False, type=bool,
                        help='save model checkpoints every 10 epochs')
    parser.add_argument('-fr', '--freeze', required=False, type=bool,
                        help='freeze layers of pre-loaded model')
    parser.add_argument('-pf', '--save_pfp', required=False, type=bool,
                        help='save protein fingerprints')

    args = parser.parse_args()

    # Create directories
    savedir=os.getcwd()
    files = [item for item in os.listdir(savedir)]
    for folder in ['Plots','Models','Results','History','Features']:
        if folder not in files:
            os.mkdir(os.path.join(savedir,folder))
    if args.epochs==0:
        train=False
    else:
        train=True

    run_model(args.input_file, savedir,args.epochs, args.results, args.output_layer,
              args.loadfile, args.f1, train, args.batch_size, args.save_checkpoints, args.freeze,
              args.learning_rate, args.save_pfp)

