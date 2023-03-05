import sys
import math
from collections import defaultdict
import time
start_time = time.time()

class Node:
    def __init__(self, word, tag, prob, total_prob, path, true_path, prob_path):
        self.word = word
        self.tag = tag
        self.prob = prob
        self.total_prob = total_prob
        self.path = path
        self.true_path = true_path
        self.prob_path = prob_path

def read_data(filename):
    data = list()
    with open(filename, 'r') as f:
        for line in f:
            if line:
                line = line.strip().split(' ')
                features = set()
                for x in range(2, len(line), 2):
                    features.add(line[x])
                data.append((line[0], line[1], features))
    return data

def read_boundary(file):
    boundaries = list()
    with open(file, 'r') as f:
        for line in f:
            if line:
                boundaries.append(int(line.strip()))
        return boundaries

def read_model(file):
    with open(file, 'r') as f:
        data = defaultdict(dict)
        labels = list()
        for line in f:
            if line:
                if 'FEATURES' in line:
                    cl = line.split()[3]
                    labels.append(cl)
                else:
                    feat = line.split()[0]
                    prob = float(line.split()[1])
                    data[cl][feat] = prob
        return data, labels

def calc_prob(features):
    probs = dict()
    for label in labels:
        exp = model[label]['<default>']
        for feat in features:
            if feat in model[label]:
                exp += model[label][feat]
        prob = math.exp(exp)
        probs[label] = prob
    z = sum(probs.values())
    for k in probs:
        probs[k] = math.log(probs[k]/z, 10)
    probs = sorted(probs.items(), key=lambda x: -x[1])[:topN]
    return probs

def prune(beamSize, topK, nodes):
    kept_nodes = list()
    max_prob = max(node.total_prob for node in nodes)
    for node in nodes:
        if node.total_prob + beamSize >= max_prob:
            kept_nodes.append(node)
    kept_nodes = sorted(kept_nodes, key=lambda x: -x.total_prob)[:topK]
    return kept_nodes

def beam_search(file):
    with open(file, 'w') as f:
        correct_count = 0
        f.write('%%%%% test data:')
        position = 0
        for i in range(len(bounds)):
            sent_len = bounds[i]
            root = Node('/s', 'BOS', 0, 0, ['BOS'], ['BOS'], [0])
            curr_nodes = [root]
            for j in range(sent_len):
                idx = position+j
                curW = data[idx][0]
                true_label = data[idx][1]
                feats = data[idx][2]
                next_nodes = list()
                for node in curr_nodes:
                    prev_prev_tag = node.path[-2] if j > 0 else 'BOS'
                    feats.add('prevT={}'.format(node.path[-1]))
                    feats.add('prevTwoTags={}+{}'.format(prev_prev_tag, node.path[-1]))
                    probs = calc_prob(feats)
                    for label, prob in probs:
                        next_nodes.append(Node(curW, label, prob, node.total_prob+prob, node.path+[label], node.true_path+[true_label], node.prob_path+[prob]))
                    feats.remove('prevT={}'.format(node.path[-1]))
                    feats.remove('prevTwoTags={}+{}'.format(prev_prev_tag, node.path[-1]))
                next_nodes = prune(beam_size, topK, next_nodes)
                curr_nodes = next_nodes
            best_end_node = next_nodes[0]
            for node in next_nodes:
                if node.total_prob > best_end_node.total_prob:
                    best_end_node = node
            for j in range(sent_len):
                idx = position + j
                f.write('\n{} {} {} {}'.format(data[idx][0], data[idx][1], best_end_node.path[j+1], math.exp(best_end_node.prob_path[j+1])))
                correct_count += data[idx][1]==best_end_node.path[j+1]
            position += sent_len
        print(correct_count/sum(bounds))

if __name__ == "__main__":
    beam_size = float(sys.argv[5])
    topN = int(sys.argv[6])
    topK = int(sys.argv[7])
    data = read_data(sys.argv[1])
    bounds = read_boundary(sys.argv[2])
    model, labels = read_model(sys.argv[3])
    beam_search(sys.argv[4])
    print("--- %s seconds ---" % (time.time() - start_time))