from glob import glob
import config
import os

train_files = glob(config.path2data + '/*-train-*')
dev_files = glob(config.path2data + '/*-dev')

for trainfile in train_files:
    lang = trainfile.split('/')[-1].split('-')[0]
    devfile = config.path2data + '/' + lang + '-dev'
    print('python3 main.py ' + trainfile + ' ' + devfile)
    os.system('python3 main.py ' + '--dynet-autobatch 1 ' + '--dynet-mem 2048 ' + trainfile + ' ' + devfile)

