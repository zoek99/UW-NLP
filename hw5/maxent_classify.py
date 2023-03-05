import sys
import math
import numpy as np
from collections import defaultdict

test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

def toidx(lst):
    lst2idx = defaultdict(int)
    idx2lst = dict()
    idx = 0
    for cl in lst:
        lst2idx[cl] = idx
        idx += 1
    for cl in lst2idx:
        idx2lst[lst2idx[cl]] = cl
    return lst2idx, idx2lst

def load_data(file):
    with open(file, 'r') as f:
        data = defaultdict(dict)
        classes = list()
        for line in f:
            if line:
                if 'FEATURES' in line:
                    cl = line.split()[3]
                    classes.append(cl)
                else:
                    feat = line.split()[0]
                    prob = float(line.split()[1])
                    data[cl][feat] = prob
        return data, classes

def load_test(file):
    feat_dict = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            if line:
                line = line.split()
                feats = line[1:]
                doc_feat_dict = dict()
                for feat in feats:
                    doc_feat_dict[feat.split(':')[0]] = int(feat.split(':')[1])
                feat_dict[line[0]].append(doc_feat_dict)
    return feat_dict

def apply_model(test_data):
    cm = np.zeros((len(classes), len(classes)))
    output = list()
    for cl in classes:
        docs = test_data[cl]
        for doc in docs:
            true_label = cl
            probs = list()
            for l in data:
                e = data[l]['<default>']
                for feat in doc:
                    e += data[l][feat]
                prob = math.exp(e)
                probs.append((l, prob))
            probs = sorted([[l,p] for (l,p) in probs], key=lambda x: x[1], reverse=True)
            z = sum(tup[1] for tup in probs)
            for tup in probs:
                tup[1] = round(tup[1]/z, 5)
            output.append((true_label, probs))
            predict = probs[0][0]
            cm[class2idx[true_label], class2idx[predict]] += 1
    return cm, output

def write_output(file):
    with open(file, 'w') as f:
        f.write('%%%%% test data:')
        count = 0
        for doc in output:
            label = doc[0]
            probs = doc[1]
            f.write('\narray:{} {}'.format(count, label))
            for prob in probs:
                f.write(' {} {}'.format(prob[0], prob[1]))
            count += 1

def print_acc():
    print('Confusion matrix for the testing data:\nrow is the truth, column is the system output\n')
    print("             " + " ".join(classes))
    for cl in classes:
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in cm[class2idx[cl], :]))
    print(' Training accuracy={}\n'.format(np.trace(cm) / np.sum(cm)))

if __name__ == "__main__":
    data, classes = load_data(model_file)
    test_data = load_test(test_file)
    class2idx, idx2class = toidx(classes)
    cm, output = apply_model(test_data)
    write_output(output_file)
    print_acc()
