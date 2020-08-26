# Import libraries

import pickle,argparse,os
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

# Map from onehot encoding to index

def get_label(onehot):
    out = []
    for l in range(len(onehot)):
        idx = np.argmax(onehot[l], axis=-1)
        out.append(idx)
    return out

def process_labels(labels):
    original=[]

    for i,l in enumerate(labels):
        processed_labels=get_label(l)
        original.append(processed_labels)

    return original

# Convert indices to original label

def get_reverse_mapping(onehotdict_path):
    dicts = []
    with open(onehotdict_path, 'rb') as f:
        onehotdict = pickle.load(f)
    for key in onehotdict.keys():
        dicts.append({v: k for k, v in onehotdict[key].items()})
    return dicts

def convert(labels,dicts):
    out=[]
    for i,label in enumerate(labels):
        dict=dicts[i]
        out.append([dict[x] for x in label])
    return out


def cluster(feat,labels,num_tasks=False):
    out=[]
    if not num_tasks:
        num_tasks=2
    names = ['Class', 'Architecture', 'Topology', 'Homologous Superfamily']
    print('Computing homogeneity score across %s instances:'%(len(feat)))
    for task in range(num_tasks):
        # label = get_label(labels[task])
        est = KMeans(init='k-means++', n_clusters=len(np.unique(labels[task])), n_init=10)
        est.fit(feat, labels[task])
        print('{}: {:.2f}%'.format(names[task], metrics.homogeneity_score(labels[task], est.labels_) * 100))
        if task<2:
            out.append(est.cluster_centers_)
    return out

def add_centroids(feat,true_labels,predicted_labels,name=False,num_tasks=False):
    clusters = cluster(feat, true_labels,num_tasks)
    feat = np.concatenate([feat, clusters[0]])
    if not name:
        name=''

    for i in range(4):
        true_labels[i] = np.concatenate([true_labels[i], ['centroid_C_'+name] * len(clusters[0])])
        predicted_labels[i] = np.concatenate([predicted_labels[i], ['centroid_C'] * len(clusters[0])])

    return feat,true_labels,predicted_labels

def process_features_labels(input_npz,dicts,centroids=False,name=None,num_tasks=False):
    print('\nLoading from',input_npz)
    upload = np.load(input_npz, allow_pickle=True)['features_labels_predicted']
    feat = upload[0]
    labels = upload[1]
    ypred = upload[2]

    original=process_labels(labels)
    converted=convert(original,dicts)
    predicted=process_labels(ypred)
    converted_ypred=convert(predicted,dicts)

    if centroids:
        print('Adding centroids')
        feat, converted, converted_ypred = add_centroids(feat, converted, converted_ypred,name,num_tasks)

    return feat,converted,converted_ypred

def tSNE(features):
    print('\nApplying t-SNE to %s vectors' % (len(features)))
    tsne = TSNE(perplexity=3, n_components=2, init='pca', n_iter=250, method='exact')
    np.set_printoptions(suppress=True)
    T = pd.DataFrame(tsne.fit_transform(features))  # Transform feature vectors into 2D
    return T


def get_featuremap(T,true_labels,predicted_labels, save_directory, descriptor,print_testplots=False):

    names = ['Class', 'Architecture', 'Topology', 'Homology']
    print('\nTrue labels')

    # Map t-SNE transform to true labels

    for i in range(len(true_labels)):

        name = names[i]
        print(name)

        T['original'] = true_labels[i]
        plt.figure(figsize=(14, 8))
        for l in np.unique(T['original'].values):
            subset = T[T['original'] == l]
            if 'centroid' in l:
                plt.scatter(subset.loc[:, 0], subset.loc[:, 1], marker='*',label=l, s=10)
            else:
                plt.scatter(subset.loc[:, 0], subset.loc[:, 1], label=l,s=10)
        if i <= 1:
            plt.legend(loc='right', title=name, bbox_to_anchor=(1.1, 0.5))
        plt.title('T-SNE Transform of Feature Map, True {}'.format(name))
        plt.savefig(save_directory + '/{}_{}_features.png'.format(descriptor, name))

    # Map t-SNE transform to predicted labels
    if print_testplots:
        print('\nPredicted labels')
        for i in range(len(predicted_labels)):
            name = names[i]
            print(name)
            T['predicted'] = predicted_labels[i]
            plt.figure(figsize=(14, 8))
            for l in np.unique(T['predicted'].values):
                subset = T[T['predicted'] == l]
                if l == 'centroid_C':
                    plt.scatter(subset.loc[:, 0], subset.loc[:, 1], marker='*', label=l, s=10)
                else:
                    plt.scatter(subset.loc[:, 0], subset.loc[:, 1], label=l,s=10)
            if i <= 1:
                plt.legend(loc='right', title=name, bbox_to_anchor=(1.1, 0.5))
            plt.title('T-SNE Transform of Feature Map, Predicted {}'.format(name))
            plt.savefig(save_directory + '/{}_{}_features_pred.png'.format(descriptor, name))

    print('Complete')

if __name__ == '__main__':
    # Handle command line options
    parser = argparse.ArgumentParser(description='Get plots of PFP clusters')
    parser.add_argument('-i', '--input_file', required=True,
                        help='Target .npz file')
    parser.add_argument('-n', '--num_tasks', type=int,required=False,
                        help='Set number of tasks for computation of homogeneity score')
    parser.add_argument('-gp', '--get_predicted', required=False,
                        help='Save plot of predicted labels')
    args = parser.parse_args()

    input_npz=args.input_file

    onehotdict_path = 'CATHonehot.pickle'
    save_directory = 'Plots/'
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

    get_testplots=args.get_predicted
    name=False
    if not args.num_tasks:
        num_tasks=2
    else:
        num_tasks=args.num_tasks

    dicts = get_reverse_mapping(onehotdict_path)
    feat, converted, converted_ypred = process_features_labels(input_npz, dicts, True,name,num_tasks)
    tsne=tSNE(feat)
    descriptor = input_npz.split('/')[-1][:-4]
    get_featuremap(tsne, converted, converted_ypred,save_directory,descriptor,get_testplots)

    # Features/20200825_1423_densenet121_HRCA_TRAINTEST_PFP.npz
