import sys
import math
import numpy as np
from collections import defaultdict, Counter



train_file = sys.argv[1]
test_file = sys.argv[2]
k_val = int(sys.argv[3])
sim_func_id = int(sys.argv[4])
output_file = sys.argv[5]



def load_data(file):
    classes = list()
    feat_dict = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            if line:
                line = line.split()
                if line[0] not in classes:
                    classes.append(line[0])
                feats = line[1:]
                doc_feat_dict = dict()
                for feat in feats:
                    doc_feat_dict[feat.split(':')[0]] = int(feat.split(':')[1])
                feat_dict[line[0]].append(doc_feat_dict)
        class2idx = defaultdict(int)
        idx2class = defaultdict(str)
        idx = 0
        for label in classes:
            class2idx[label] = idx
            idx += 1
        for k, v in class2idx.items():
            idx2class[v] = k
    return feat_dict, class2idx, idx2class, classes



def cosine(feats1, feats2):
    dot_prod = sum(v * feats2.get(k, 0) for k,v in feats1.items())
    magA = math.sqrt(sum(v ** 2 for v in feats1.values()))
    magB = math.sqrt(sum(v ** 2 for v in feats2.values()))
    return dot_prod / (magA * magB)



def euclidean(feats1, feats2):
    return math.sqrt(sum((v - feats2.get(k, 0)) ** 2 for k, v in feats1.items()))



def knn(file, sim_func_id, k_val):
    feat_dict, class2idx, idx2class, classes = load_data(file)
    test_feat_dict, class2idx1, idx2class1, classes1 = load_data(test_file)
    output = list()
    test_output = list()
    cm = np.zeros((len(classes), len(classes)))
    test_cm = np.zeros((len(classes), len(classes)))
    if sim_func_id == 1: #euclidean
        for label in classes:
            docs = feat_dict[label]
            test_docs = test_feat_dict[label]
            for doc in docs:
                neighbors = list()
                res = list()
                res.append(label)
                for k, v in feat_dict.items():
                    for doc2 in v:
                        distance = euclidean(doc, doc2)
                        if len(neighbors) < k_val:
                            neighbors.append((distance, k))
                            # low to high
                            neighbors = sorted(neighbors, key = lambda x: x[0])
                        else:
                            if distance < neighbors[-1][0]:
                                neighbors[-1] = (distance, k)
                                neighbors = sorted(neighbors, key=lambda x: x[0])
                votes = Counter([k for (d, k) in neighbors])
                for cl in classes:
                    if cl not in votes:
                        votes[cl] = 0.0
                for k in votes:
                    votes[k] = round(votes[k] / k_val, 5)
                votes = sorted([(k,v) for k,v in votes.items()], key=lambda x: x[1], reverse=True)
                res.append(votes)
                cm[class2idx[res[1][0][0]], class2idx[res[0]]] += 1
                output.append(res)
            for doc in test_docs:
                neighbors = list()
                res = list()
                res.append(label)
                for k, v in feat_dict.items():
                    for doc2 in v:
                        distance = euclidean(doc, doc2) #should be similarity
                        if len(neighbors) < k_val:
                            neighbors.append((distance, k))
                            # low to high
                            neighbors = sorted(neighbors, key = lambda x: x[0])
                        else:
                            if distance < neighbors[-1][0]:
                                neighbors[-1] = (distance, k)
                                neighbors = sorted(neighbors, key=lambda x: x[0])
                votes = Counter([k for (d, k) in neighbors])
                for cl in classes:
                    if cl not in votes:
                        votes[cl] = 0.0
                for k in votes:
                    votes[k] = round(votes[k] / k_val, 5)
                votes = sorted([(k,v) for k,v in votes.items()], key=lambda x: x[1], reverse=True)
                res.append(votes)
                test_cm[class2idx[res[1][0][0]], class2idx[res[0]]] += 1
                test_output.append(res)
        return output, test_output, cm, test_cm, class2idx, idx2class, classes

    if sim_func_id == 2: #cosine
        for label in classes:
            docs = feat_dict[label]
            test_docs = test_feat_dict[label]
            for doc in docs:
                neighbors = list()
                res = list()
                res.append(label)
                for k, v in feat_dict.items():
                    for doc2 in v:
                        distance = cosine(doc, doc2)
                        if len(neighbors) < k_val:
                            neighbors.append((distance, k))
                            # high to low
                            neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)
                        else:
                            if distance > neighbors[-1][0]:
                                neighbors[-1] = (distance, k)
                                neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)
                votes = Counter([k for (d, k) in neighbors])
                for cl in classes:
                    if cl not in votes:
                        votes[cl] = 0.0
                for k in votes:
                    votes[k] = round(votes[k] / k_val, 5)
                votes = sorted([(k,v) for k,v in votes.items()], key=lambda x: x[1], reverse=True)
                res.append(votes)
                cm[class2idx[res[1][0][0]], class2idx[res[0]]] += 1
                output.append(res)
            for doc in test_docs:
                neighbors = list()
                res = list()
                res.append(label)
                for k, v in feat_dict.items():
                    for doc2 in v:
                        distance = cosine(doc, doc2)
                        if len(neighbors) < k_val:
                            neighbors.append((distance, k))
                            # high to low
                            neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)
                        else:
                            if distance > neighbors[-1][0]:
                                neighbors[-1] = (distance, k)
                                neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)
                votes = Counter([k for (d, k) in neighbors])
                for cl in classes:
                    if cl not in votes:
                        votes[cl] = 0.0
                for k in votes:
                    votes[k] = round(votes[k] / k_val, 5)
                votes = sorted([(k,v) for k,v in votes.items()], key=lambda x: x[1], reverse=True)
                res.append(votes)
                test_cm[class2idx[res[1][0][0]], class2idx[res[0]]] += 1
                test_output.append(res)
        return output, test_output, cm, test_cm, class2idx, idx2class, classes



if __name__ == "__main__":
    train_output, test_output, train_cm, test_cm, class2idx, idx2class, classes = knn(train_file, sim_func_id, k_val)
    with open(output_file, 'w') as f:
        f.write('%%%%% training data:\n')
        count = 0
        for res in train_output:
            f.write('array:{} {} {} {} {} {} {} {}\n'.format(count, res[0], res[1][0][0], res[1][0][1], res[1][1][0], res[1][1][1], res[1][2][0], res[1][2][1]))
            count += 1
        f.write('%%%%% testing data:')
        for res in test_output:
            f.write('\narray:{} {} {} {} {} {} {} {}'.format(count, res[0], res[1][0][0], res[1][0][1], res[1][1][0], res[1][1][1], res[1][2][0], res[1][2][1]))
            count += 1
    print('Confusion matrix for the training data:\nrow is the truth, column is the system output\n')
    print("             " + " ".join(classes))
    for cl in classes:
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in train_cm[class2idx[cl], :]))
    print(' Training accuracy={}\n'.format(np.trace(train_cm) / np.sum(train_cm)))
    print('Confusion matrix for the test data:\nrow is the truth, column is the system output\n')
    print("             " + " ".join(idx2class.values()))
    for cl in idx2class.values():
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in test_cm[class2idx[cl], :]))
    print(' Test accuracy={}'.format(np.trace(test_cm) / np.sum(test_cm)))
