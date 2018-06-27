from preprocess import Vocab
import config

data = []

with open(config.PATH2DATA, 'r') as file:
    line = file.readline()
    while line:
        s, t, f = line.split()
        data.append((s, t, f.split(';')))
        line = file.readline()
