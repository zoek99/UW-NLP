import sys
import math

def read_model(file):
    with open(file, 'r') as f:
        model = dict()
        weights = list()
        sv = list()
        all_feats = set()
        for line in f:
            line = line.strip().split()
            if line[0] == 'svm_type' or line[0] == 'nr_class' or line[0] == 'nr_sv' or line[0] == 'SV' or line[0] == 'total_sv' or line[0] == 'label':
                pass
            elif line[0] == 'kernel_type':
                model['kernel_type'] = line[1]
            elif line[0] == 'rho':
                model['rho'] = float(line[1])
            elif line[0] == 'gamma':
                model['gamma'] = float(line[1])
            elif line[0] == 'coef0':
                model['coef0'] = float(line[1])
            elif line[0] == 'degree':
                model['degree'] = int(line[1])
            else:
                weights.append(float(line[0]))
                vector = set()
                for feat in line[1:]:
                    feat = feat.split(':')[0]
                    all_feats.add(feat)
                    vector.add(feat)
                sv.append(vector)
    return model, weights, sv, all_feats

def read_data(file):
    with open(file, 'r') as f:
        data = list()
        for line in f:
            line = line.strip().split()
            feats = set()
            for feat in line[1:]:
                feats.add(feat.split(':')[0])
            data.append((line[0], feats))
    return data

def kernel_func(v1, v2):
    # use set operations here to replace inner product (intersection and symmetric difference)
    # since all features are either present or not, the value(count) does not matter
    if model['kernel_type'] == 'linear':
        return len(v1 & v2)
    elif model['kernel_type'] == 'polynomial':
        return (model['gamma'] * len(v1 & v2) + model['coef0']) ** model['degree']
    elif model['kernel_type'] == 'rbf':
        return math.exp(-model['gamma'] * len(v1.symmetric_difference(v2)))
    elif model['kernel_type'] == 'sigmoid':
        return math.tanh(model['gamma']*len(v1 & v2) + model['coef0'])

def decode(file):
    with open(file, 'w') as f:
        correct_count = 0
        for i in range(len(data)):
            true = data[i][0]
            v1 = data[i][1]
            fx = 0
            for j in range(len(sv)):
                v2 = sv[j]
                fx += (weights[j] * kernel_func(v1,v2))
            fx -= model['rho']
            if fx >= 0:
                predict = '0'
            else:
                predict = '1'
            if true == predict:
                correct_count += 1
            f.write('{} {} {}\n'.format(true, predict, fx))
        print('test acc: {}'.format(correct_count/len(data)))

if __name__ == "__main__":
    model, weights, sv, all_feats = read_model(sys.argv[2])
    data = read_data(sys.argv[1])
    decode(sys.argv[3])