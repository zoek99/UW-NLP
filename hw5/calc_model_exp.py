import sys
import math
from collections import defaultdict, Counter

def load_model(file):
    with open(file, 'r') as f:
        data = defaultdict(dict)
        for line in f:
            if line:
                if 'FEATURES' in line:
                    cl = line.split()[3]
                else:
                    feat = line.split()[0]
                    prob = float(line.split()[1])
                    data[cl][feat] = prob
        return data

def load_train(file):
    with open(file, 'r') as f:
        classes = list()
        feat_dict = defaultdict(list)
        total = 0
        for line in f:
            if line:
                line = line.strip().split()
                label = line[0]
                if label not in classes:
                    classes.append(label)
                feats = line[1:]
                doc_feat_dict = dict()
                for feat in feats:
                    doc_feat_dict[feat.split(':')[0]] = int(feat.split(':')[1])
                feat_dict[line[0]].append(doc_feat_dict)
                total += 1
        return classes, feat_dict, total

def calc_exp(data):
    counts = defaultdict(Counter)
    for cl in classes:
        docs = data[cl]
        for feats in docs:
            if len(sys.argv) == 4:
                prob_tup = list()
                for l in classes:
                    e = model_data[l]['<default>']
                    for feat in feats:
                        e += model_data[l][feat]
                    prob = math.exp(e)
                    prob_tup.append((l, prob))
                z = sum(tup[1] for tup in prob_tup)
                label_prob = dict()
                for tup in prob_tup:
                    label_prob[tup[0]] = tup[1] / z
                for feat in feats:
                    for l1 in classes:
                        counts[feat][l1] += label_prob[l1]
            else:
                c = len(classes)
                for feat in feats:
                    for l1 in classes:
                        counts[feat][l1] += 1/c
    return counts

def write_output(file):
    with open(file, 'w') as f:
        for cl in classes:
            for feat in sorted(counts.keys()):
                exp = round(counts[feat][cl] / total, 5)
                count = round(counts[feat][cl], 5)
                f.write('{} {} {} {}\n'.format(cl, feat, exp, count))

if __name__ == "__main__":
    classes, feat_dict, total = load_train(sys.argv[1])
    if len(sys.argv) == 3:
        counts = calc_exp(feat_dict)
    elif len(sys.argv) == 4:
        model_data = load_model(sys.argv[3])
        counts = calc_exp(feat_dict)
    write_output(sys.argv[2])
