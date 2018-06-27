path2data = '/Users/tomoki/NLP_data/conll2018/task1/all/'
trainfile = path2data + 'portuguese-train-medium'
parsefile = path2data + 'portuguese-dev'
savedir = './experiments0627/'

epochs = 100
quit_after_n_epochs_without_update = 10
batch_size = 10
adam = True

layers = 1
pdrop_embs = 0.2
pdrop_mlp = 0.2
pdrop_lstm = 0.2

learning_rate = 0.002
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-12
clip_threshold = 1.0
decay = 0.9
decay_steps = 5000



char_dim = 100
feat_dim = 50
hidden_dim = 100
