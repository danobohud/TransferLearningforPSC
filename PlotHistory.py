import pickle,argparse
import numpy as np
import matplotlib.pyplot as plt
from Helpers import load_pickle

def load_(path):
    hist = load_pickle(path)
    outdict = {}
    for key in hist.keys():
        try:
            newkey = key.decode('utf8')
            outdict[newkey] = hist[key]
        except AttributeError:
            outdict[key] = hist[key]
    return outdict

def training_plot(hist):
    (fig, ax) = plt.subplots(4, 1, figsize=(10, 10))
    for i, task in enumerate(['C', 'A', 'T', 'H']):
        train = '%s_labels_accuracy' % (task)
        val = 'val_' + train
        y_train = hist[train]
        y_val = hist[val]
        x = np.arange(1, len(y_train) + 1)
        ax[i].plot(x, y_train, label='train')
        ax[i].plot(x, y_val, label='val')
        ax[i].set_title('Task %s' % (task))
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel('Accuracy (%)')
        ax[i].legend()
    fig.tight_layout(pad=1.5)
    plt.show()
    plt.clf()

def training_plot_binary(hist):
    (fig, ax) = plt.subplots(1, 1, figsize=(10, 10))
    train = 'accuracy'
    val = 'val_' + train
    y_train = hist[train]
    y_val = hist[val]
    x = np.arange(1, len(y_train) + 1)
    plt.plot(x, y_train, label='train')
    plt.plot(x, y_val, label='val')
    plt.title('Training plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
    plt.clf()

def print_max(hist):
    print('Maximum training and validation accuracies:\n')
    for i, task in enumerate(['C', 'A', 'T', 'H']):
        train = '%s_labels_accuracy' % (task)
        val = 'val_' + train
        y_train = hist[train]
        y_val = hist[val]
        print('{}: train = {:.1f}%, val = {:.1f}% @ epoch {}'.format(task,
                                                                     np.max(y_train) * 100, np.max(y_val) * 100,
                                                                     y_train.index(np.max(y_train))))

if __name__ == '__main__':
    # Handle command line options
    parser = argparse.ArgumentParser(description='Get training plots and maximum accuracies from model history .pickle files')
    parser.add_argument('-i', '--input_file', required=True,
                        help='Target .pickle file')
    args = parser.parse_args()
    hist = load_pickle(args.input_file)
    training_plot(hist)
    print_max(hist)
