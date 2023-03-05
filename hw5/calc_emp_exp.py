import sys
from collections import defaultdict, Counter

def load_data(file):
    with open(file, 'r') as f:
        classes = list()
        data = defaultdict(Counter)
        total = 0
        for line in f:
            if line:
                line = line.strip().split()
                label = line[0]
                if label not in classes:
                    classes.append(label)
                feats = line[1:]
                for feat in feats:
                    name = feat.split(':')[0]
                    data[name][label] += 1
                total += 1
        return classes, data, total

def write_output(file):
    with open(file, 'w') as f:
        for cl in classes:
            for feat in sorted(data.keys()):
                exp = round(data[feat][cl] / total, 5)
                count = data[feat][cl]
                f.write('{} {} {} {}\n'.format(cl, feat, exp, count))

if __name__ == "__main__":
    classes, data, total = load_data(sys.argv[1])
    write_output(sys.argv[2])