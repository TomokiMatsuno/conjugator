import dynet as dy
import numpy as np

from preprocess import Vocab

import config
import model

data = [[], []]

with open(config.trainfile, 'r') as file:
    line = file.readline()
    while line:
        s, t, f = line.split('\t')
        data[0].append((s, t, f.split(';')))
        line = file.readline()
with open(config.parsefile, 'r') as file:
    line = file.readline()
    while line:
        s, t, f = line.split('\t')
        data[1].append((s, t, f.split(';')))
        line = file.readline()


vocab = Vocab(data[0])
vocab.add_parsefile(data[1])

mdl = model.Model(char_dim=config.char_dim, feat_dim=config.feat_dim, hidden_dim=config.hidden_dim,
                  char_size=len(vocab._char_dict.x2i), feat_sizes=[len(fd.x2i) for fd in vocab._feat_dicts])

for epc in range(config.epochs):
    for step in range(10):
        step = step == 9
        losses = []
        tot_cor = 0
        tot_loss = 0
        isTrain = (1 - step)
        ids = [i for i in range(len(data[step]))]
        if isTrain:
            np.random.shuffle(ids)

        for i in ids:
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
                if len(pred_word) < 20:
                    print(d[1], pred_word)

        if not isTrain:
            print('accuracy:', tot_cor / len(data[step]))
        else:
            print('loss:', tot_loss)







