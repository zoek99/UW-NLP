import sys
from collections import defaultdict
import math
import numpy as np

train_file = sys.argv[1]
test_file = sys.argv[2]
max_depth = int(sys.argv[3])
min_gain = float(sys.argv[4])
model_file = sys.argv[5]
sys_output = sys.argv[6]


class Node:
    def __init__(self, indices, depth, entropy, exclude, path, feat = "", left = None, right = None):
        self.feat = feat
        self.left = left
        # left = present feature
        self.right = right
        self.indices = indices
        # {class:(idx)}
        self.entropy = entropy
        self.depth = depth
        self.exclude = exclude
        self.indices_count = sum(len(v) for k,v in indices.items())
        self.prob = None
        self.path = path


    def get_prob(self, all_classes):
        if self.prob is not None:
            return self.prob
        self.prob = list()
        for cl in all_classes:
            self.prob.append((cl, len(self.indices[cl])/self.indices_count))
        return self.prob


def calc_entropy(indices):
    h = 0
    total = 0
    for k,v in indices.items():
        total += len(v)
    for k,v in indices.items():
        prob = len(v)/total
        if prob != 0:
            h -= prob*math.log(prob,2)
    return h


def load_data(file):
    """Return {class1: [(f1,f2..),(f1..),..]
               class2: []}"""
    classes = defaultdict(list)
    feats = set()
    feed_to_model = list()
    with open(file, 'r') as f:
        for line in f:
            if line:
                line = line.strip().split()
                items = set()
                lst = list()
                lst.append(line[0])
                feed_to_model.append(lst)
                for i in range(1, len(line)):
                    items.add(line[i].split(':')[0])
                    feats.add(line[i].split(':')[0])
                classes[line[0]].append(items)
                feed_to_model[-1].append(items)
    return classes, feats, feed_to_model


def split(classes, feats, node, min_gain, max_depth):
    if node.depth > max_depth:
        return
    base = min_gain
    best_feat = ""
    best_left_indices = defaultdict(set)
    best_right_indices = defaultdict(set)
    left_entropy = 0
    right_entropy = 0
    for feat in feats:
        if feat not in node.exclude:
            mi = node.entropy
            left_indices = defaultdict(set)
            right_indices = defaultdict(set)
            for cl, idx in node.indices.items():
                for i in idx:
                    doc = classes[cl][i]
                    if feat in doc:
                        left_indices[cl].add(i)
                    else:
                        right_indices[cl].add(i)
            prob = sum(len(v) for v in left_indices.values()) / node.indices_count
            left_entropy = calc_entropy(left_indices)
            right_entropy = calc_entropy(right_indices)
            mi = mi - prob * left_entropy - (1-prob) * right_entropy
            if mi > base:
                base = mi
                best_feat = feat
                best_left_indices = left_indices
                best_right_indices = right_indices
    if best_feat != "":
        node.feat = best_feat
        node.exclude.add(best_feat)
        node.left = Node(best_left_indices, node.depth + 1, left_entropy, {s for s in node.exclude}, node.feat if node.path == "" else node.path+'&'+node.feat)
        node.right = Node(best_right_indices, node.depth + 1, right_entropy, {s for s in node.exclude}, "!"+node.feat if node.path == "" else node.path+'&!'+node.feat)


def build_dt(classes, feats, min_gain, max_depth):
    model = []
    indices = defaultdict(set)
    for k,v in classes.items():
        indices[k] = {i for i in range(len(v))}
    root = Node(indices, 0, calc_entropy(indices), set(), '')
    to_split = [root]
    while len(to_split) != 0:
        times = len(to_split)
        for i in range(times):
            node = to_split.pop(0)
            split(classes, feats, node, min_gain, max_depth)
            if node.left is not None and len(node.left.indices) > 0:
                to_split.append(node.left)
            if node.right is not None and len(node.right.indices) > 0:
                to_split.append(node.right)
            if node.right is None and node.left is None:
                model_item = ''
                model_item += '{} {}'.format(node.path, node.indices_count)
                for prob in node.get_prob(list(classes.keys())):
                    model_item += ' {} {}'.format(prob[0], prob[1])
                model.append(model_item)
    return root, model


def output(classes, file, root, fsys):
    cm = np.zeros((len(classes), len(classes)))
    class2idx = defaultdict(int)
    idx2class = defaultdict(str)
    idx = 0
    for k,v in classes.items():
        class2idx[k] = idx
        idx += 1
    for k,v in class2idx.items():
        idx2class[v] = k
    for i in range(len(file)):
        cl = file[i][0]
        node = root
        parent = root
        while node:
            if node.feat in file[i][1]:
                parent = node
                node = node.left
            else:
                parent = node
                node = node.right
        fsys.write('array:{}'.format(i))
        probs = parent.get_prob(list(idx2class.values()))
        base = 0
        label = ''
        for prob in probs:
            fsys.write(' {} {}'.format(prob[0], prob[1]))
            if prob[1]>base:
                label = prob[0]
                base = prob[1]
        fsys.write('\n')
        cm[class2idx[cl], class2idx[label]] += 1
    acc = np.trace(cm)/np.sum(cm)
    print(" ".join(idx2class.values()))
    # for cl in idx2class.values():
    #     print(' {}'.format(cl))
    for cl in idx2class.values():
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in cm[class2idx[cl], :]))
    print('\n')
    return cm, acc


if __name__ == "__main__":
    classes, feats, train = load_data(train_file)
    classes_test, feats_test, test = load_data(test_file)
    root, model = build_dt(classes, feats, min_gain, max_depth)
    with open(sys_output, 'w') as f:
        f.write('%%%%% training data:')
        print('Confusion matrix for the training data:\nrow is the truth, column is the system output\n             ', end='')
        train_cm, train_acc = output(classes, train, root, f)
        print(' Training accuracy={}\n\n'.format(train_acc))
        print('Confusion matrix for the test data:\nrow is the truth, column is the system output\n             ', end='')
        f.write('\n%%%%% test data:')
        test_cm, test_acc = output(classes_test, test, root, f)
        print(' Test accuracy={}'.format(test_acc))
    with open(model_file, 'w') as f:
        for line in model:
            f.write('{}\n'.format(line))



