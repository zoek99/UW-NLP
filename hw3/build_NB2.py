import sys
import math
from collections import defaultdict
import numpy as np

train_file = sys.argv[1]
test_file = sys.argv[2]
class_prior_delta = float(sys.argv[3])
cond_prob_delta = float(sys.argv[4])
model_file = sys.argv[5]
sys_output = sys.argv[6]

def load_data(file):
    classes = list()
    feat_dict = defaultdict(list)
    word_count_dict = defaultdict(list)
    vocab = set()
    z = defaultdict(int)
    cnt = defaultdict(int)
    with open(file, 'r') as f:
        for line in f:
            if line:
                count = 0
                line = line.split()
                if line[0] not in classes:
                    classes.append(line[0])
                feats = line[1:]
                doc_feat_dict = dict()
                for feat in feats:
                    doc_feat_dict[feat.split(':')[0]] = float(feat.split(':')[1])
                    count += int(feat.split(':')[1])
                    vocab.add(feat.split(':')[0])
                    cnt[(feat.split(':')[0], line[0])] += int(feat.split(':')[1])
                    z[line[0]] += int(feat.split(':')[1])
                feat_dict[line[0]].append(doc_feat_dict)
                word_count_dict[line[0]].append(count)
        class2idx = defaultdict(int)
        idx2class = defaultdict(str)
        idx = 0
        for label in classes:
            class2idx[label] = idx
            idx += 1
        for k, v in class2idx.items():
            idx2class[v] = k
    return feat_dict, word_count_dict, vocab, z, cnt, class2idx, idx2class, classes



def train_multinomial(feat_dict, word_count_dict, vocab, z, cnt):
    class_prob = defaultdict(tuple)
    cond_prob = defaultdict(dict)
    # calculate class prior probability
    class_count = 0
    for label in feat_dict:
        class_count += len(feat_dict[label])
    for label in feat_dict:
        count = len(feat_dict[label])
        prob = (class_prior_delta + count)/(len(feat_dict)*class_prior_delta + class_count)
        class_prob[label] = (prob, math.log(prob, 10))
    # calculate conditional probability p(w|c)
    for label in feat_dict:
        for feat in vocab:
            prob = (cond_prob_delta + cnt[(feat, label)])/(cond_prob_delta*len(vocab) + z[label])
            cond_prob[label][feat] = (prob, math.log(prob, 10))
    return class_prob, cond_prob



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



def apply_model(cond_prob, file, class_prob):
    feat_dict, word_count_dict, vocab, z, cnt, class2idx, idx2class, classes = load_data(file)
    output = list()
    # confusion matrix
    cm = np.zeros((len(classes), len(classes)))
    # make predictions
    for true_label in classes:
        docs = feat_dict[true_label]
        for doc in docs:
            item = [true_label]
            # probs here are in log
            probs = defaultdict(float)
            for feat in doc:
                for label in z:
                    if feat in cond_prob[label]:
                        probs[label] += doc[feat]*cond_prob[label][feat][1]
            for label1 in z:
                probs[label1] += class_prob[label1][1]
            x = -max(probs.values())
            prob_doc = sum(10 ** (value+x) for value in probs.values())
            predict = true_label
            base = 0
            for label2, prob in probs.items():
                prob = 10**(prob+x) / prob_doc
                item.append((label2, prob))
                if prob > base:
                    predict = label2
                    base = prob
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
    feat_dict, word_count_dict, vocab, z, cnt, class2idx, idx2class, classes = load_data(train_file)
    class_prob, cond_prob = train_multinomial(feat_dict, word_count_dict, vocab, z, cnt)
    write_model(class_prob, cond_prob, model_file)
    train_output, train_cm = apply_model(cond_prob, train_file, class_prob)
    test_output, test_cm= apply_model(cond_prob, test_file, class_prob)
    write_output(train_output, test_output, sys_output)
    print('Confusion matrix for the training data:\nrow is the truth, column is the system output\n')
    print("             "+" ".join(idx2class.values()))
    for cl in idx2class.values():
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in train_cm[class2idx[cl], :]))
    print('\n Training accuracy={}\n\n'.format(np.trace(train_cm) / np.sum(train_cm)))
    print('Confusion matrix for the test data:\nrow is the truth, column is the system output\n')
    print("             "+" ".join(idx2class.values()))
    for cl in idx2class.values():
        print('{} '.format(cl), end="")
        print(' '.join(str(int(val)) for val in test_cm[class2idx[cl], :]))
    print('\n Test accuracy={}'.format(np.trace(test_cm) / np.sum(test_cm)))