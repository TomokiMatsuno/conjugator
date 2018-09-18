import dynet as dy
import numpy as np
import argparse
import os

from preprocess import Vocab

import config
import model

# Iterate through files and receive arguments.
argparser = argparse.ArgumentParser()
argparser.add_argument("trainfile")
argparser.add_argument("parsefile")
argparser.add_argument("--dynet-autobatch")
argparser.add_argument("--dynet-mem")
args = argparser.parse_args()
trainfile, parsefile = args.trainfile, args.parsefile

# Create a directory for each file.
lang = trainfile.split('/')[-1].split('-')[0]
size = trainfile.split('/')[-1].split('-')[2]
save_dir = '/saves/' + lang + '-' + size + '/'
if not os.path.exists(os.curdir + save_dir):
    os.mkdir(os.curdir + save_dir)
else:
    print('the specified directory already exists. Press Y to proceed.')
    while True:
        if input() == 'Y':
            break

# Load sentences from training file and validation file.
data = [[], []]

with open(trainfile, 'r') as file:
    line = file.readline()
    while line:
        s, t, f = line.split('\t')
        data[0].append((s, t, f.split(';')))
        line = file.readline()
with open(parsefile, 'r') as file:
    line = file.readline()
    while line:
        s, t, f = line.split('\t')
        data[1].append((s, t, f.split(';')))
        line = file.readline()

# Prepare vocabulary. Here, it doesn't add words that appear only in evaluation set.
vocab = Vocab(data[0])
vocab.add_parsefile(data[1])

# Instanciate a model.
mdl = model.Model(char_dim=config.char_dim, feat_dim=config.feat_dim, hidden_dim=config.hidden_dim,
                  char_size=len(vocab._char_dict.x2i), feat_sizes=[len(fd.x2i) for fd in vocab._feat_dicts])

# Train and validate the model.
# It stops training when the maximum accuracy in validation set does not improve for more than specified epochs.
max_acc = 0
has_not_been_updated_for = 0

for epc in range(config.epochs):
    for step in range(2):
        losses = []
        tot_cor = 0
        tot_loss = 0
        isTrain = (1 - step)
        ids = [i for i in range(len(data[step]))]
        if isTrain:
            np.random.shuffle(ids)
        else:
            with open(os.curdir + save_dir + 'parsed.txt', 'w') as f:
                pass


        for i in ids:
            # Prepare a triple of the source word's character ids,
            # the target word's character ids and morphosyntactic features.
            d = data[step][i]
            triple = ([vocab._char_dict.x2i[c] for c in d[0]],
                      [vocab._char_dict.x2i[c] for c in d[1]],
                      [vocab._feat_dicts[idx].x2i[c] for idx, c in enumerate(d[2])])
            pred_word_indices, loss = mdl.run(triple, isTrain)
            losses.extend(loss)
            if isTrain:
                if len(losses) >= config.batch_size:
                    sum_loss = dy.esum(losses)
                    tot_loss += sum_loss.value()
                    sum_loss.backward()
                    mdl.update_parameters()
                    mdl._global_step += 1
                    losses = []
                    dy.renew_cg()

            else:
                pred_word = ''.join([vocab._char_dict.i2x[c] for c in pred_word_indices[:-1]])
                if pred_word == d[1]:
                    tot_cor += 1
                with open(os.curdir + save_dir + 'parsed.txt', 'a') as f:
                    f.write(d[0] + '\t' + pred_word + '\n')

        if not isTrain:
            acc = tot_cor / len(data[step])
            print('accuracy:', acc)
            if max_acc < acc:
                max_acc = acc
                has_not_been_updated_for = 0
                mdl._pc.save(os.curdir + save_dir + 'parameters')
            else:
                has_not_been_updated_for += 1
                if has_not_been_updated_for > config.quit_after_n_epochs_without_update:
                    print('accracy: ', max_acc)
                    exit(0)
        else:
            print('loss:', tot_loss)







