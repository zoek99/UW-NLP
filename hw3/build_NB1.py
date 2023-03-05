import sys
import math
from collections import defaultdict
import numpy as np



train_file = sys.argv[1]
test_file = sys.argv[2]
class_prior_delta = sys.argv[3]
cond_prob_delta = sys.argv[4]
model_file = sys.argv[5]
sys_output = sys.argv[6]



def load_data(file):
    classes = list()
    feat_dict = defaultdict(list)
    word_count_dict = defaultdict(list)
    vocab = set()
    #multinomial uncomment
    #z = defaultdict(int)
    #cnt = defaultdict(int)
    with open(file, 'r') as f:
        for line in f:
            if line:
                count = 0
                line = line.split()
                if line[0] not in classes:
                    classes.append(line[0])
                feats = line[1:]
                feat_set = set()
                for feat in feats:
                    feat_set.add(feat.split(':')[0])
                    count += int(feat.split(':')[1])
                    vocab.add(feat.split(':')[0])
                feat_dict[line[0]].append(feat_set)
                word_count_dict[line[0]].append(count)
        class2idx = defaultdict(int)
        idx2class = defaultdict(str)
        idx = 0
        for label in classes:
            class2idx[label] = idx
            idx += 1
        for k, v in class2idx.items():
            idx2class[v] = k
    return feat_dict, word_count_dict, vocab, class2idx, idx2class, classes



def train_bernoulli(feat_dict, word_count_dict, vocab, class_prior_delta, cond_prob_delta):
    class_prob = defaultdict(tuple)
    cond_prob = defaultdict(dict)
    z_c  = defaultdict(float)
    #calculate class prior probability
    class_count = 0
def load_data(file):
    classes = list()
    feat_dict = defaultdict(list)
    word_count_dict = defaultdict(list)
    vocab = set()
    #multinomial uncomment
    #z = defaultdict(int)
    #cnt = defaultdict(int)
    with open(file, 'r') as f:
        for line in f:
            if line:
                count = 0
                line = line.split()
                if line[0] not in classes:
                    classes.append(line[0])
                feats = line[1:]
                feat_set = set()
                for feat in feats:
                    feat_set.add(feat.split(':')[0])
                    count += int(feat.split(':')[1])
                    vocab.add(feat.split(':')[0])
                feat_dict[line[0]].append(feat_set)
                word_count_dict[line[0]].append(count)
        class2idx = defaultdict(int)
        idx2class = defaultdict(str)
        idx = 0
        for label in classes:
            class2idx[label] = idx
            idx += 1
        for k, v in class2idx.items():
            idx2class[v] = k
    return feat_dict, word_count_dict, vocab, class2idx, idx2class, classes
    for label in feat_dict.keys():
        class_count += len(feat_dict[label])
    for label in feat_dict.keys():
        count = len(feat_dict[label])
        prob = (class_prior_delta + count)/(len(feat_dict)*class_prior_delta + class_count)
        class_prob[label] = (prob, math.log(prob, 10))
    #calculate conditional probability
    #calculate z_c
    for label in feat_dict.keys():
        z = 1
        for feat in vocab:
            count = 0
            for doc in feat_dict[label]:
                if feat in doc:
                    count += 1
            prob = (count + cond_prob_delta)/(len(feat_dict[label]) + cond_prob_delta*2)
            z = z * (1-prob)
            cond_prob[label][feat] = (prob, math.log(prob, 10))
        z_c[label] = z
    return class_prob, cond_prob, z_c



def write_model(class_prob, cond_prob, file):
    with open(file, 'w') as f:
        f.write('%%%%% prior prob P(c) %%%%%\n')
        for label, probs in class_prob.items():
            f.write('{} {} {}\n'.format(label, probs[0], probs[1]))
        f.write('%%%%% conditional prob P(f|c) %%%%%\n')
        for label, feat_probs in cond_prob.items():
            f.write('%%%%% conditional prob P(f|c) c={} %%%%%\n'.format(label))
            for feat, probs in feat_probs.items():
                f.write('{} {} {} {}\n'.format(feat, label, probs[0], probs[1]))



def apply_model(cond_prob, z_c, file, class_prob):
    feat_dict, word_count_dict, vocab, class2idx, idx2class, classes = load_data(file)
    output = list()
    cm = np.zeros((len(classes), len(classes)))
    for true_label in classes:
        docs = feat_dict[true_label]
        for i, doc in enumerate(docs):
            item = [true_label]
            probs = defaultdict(lambda: 1.0)
            for feat in doc:
                for label in z_c:
                    if feat in cond_prob[label]:
                        probs[label] *= cond_prob[label][feat][0] / (1-cond_prob[label][feat][0])
                    # prevent underflow!
                    if probs[label] < 0.0001:
                        for l in z_c:
                            probs[l] *= 10000
            for label1 in z_c:
                probs[label1] = probs[label1] * z_c[label1] * class_prob[label1][0]
            prob_doc = sum(probs.values())
            predict = true_label
            base = 0
            for label2, prob in probs.items():
                item.append((label2, prob / prob_doc))
                if prob / prob_doc > base:
                    predict = label2
                    base = prob / prob_doc
            output.append(item)
            cm[class2idx[predict], class2idx[true_label]] += 1
    return output, cm



def write_output(train_output, test_output, output_file):
    with open(output_file, 'w') as f:
        f.write('%%%%% training data:\n')
        count = 0
        for doc in train_output:
            f.write('array:{} {}'.format(count, doc[0]))
            for class_prob_tup in doc[1:]:
                f.write(' {} {}'.format(class_prob_tup[0], class_prob_tup[1]))
            f.write('\n')
            count += 1
        f.write('%%%%% test data:\n')
        for doc in test_output:
            f.write('array:{} {}'.format(count, doc[0]))
            for class_prob_tup in doc[1:]:
                f.write(' {} {}'.format(class_prob_tup[0], class_prob_tup[1]))
            f.write('\n')
            count += 1



if __name__ == "__main__":
    feat_dict, word_count_dict, vocab, class2idx, idx2class, classes = load_data(train_file)
    class_prob, cond_prob, z_c = train_bernoulli(feat_dict, word_count_dict, vocab, float(class_prior_delta), float(cond_prob_delta))
    write_model(class_prob, cond_prob, model_file)
    train_output, train_cm = apply_model(cond_prob, z_c, train_file, class_prob)
    test_output, test_cm = apply_model(cond_prob, z_c, test_file, class_prob)
    write_output(train_output, test_output, sys_output)
    print('Confusion matrix for the training data:\nrow is the truth, column is the system output\n')
    print("             "+" ".join(idx2class.values()))
    for cl in idx2class.values():
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in train_cm[class2idx[cl], :]))
    print(' Training accuracy={}\n'.format(np.trace(train_cm)/np.sum(train_cm)))
    print('Confusion matrix for the test data:\nrow is the truth, column is the system output\n')
    print("             "+" ".join(idx2class.values()))
    for cl in idx2class.values():
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in test_cm[class2idx[cl], :]))
    print(' Test accuracy={}'.format(np.trace(test_cm)/np.sum(test_cm)))