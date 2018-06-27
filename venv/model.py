import dynet as dy
import numpy as np
import config
import utils
from preprocess import Vocab

BOW, EOW = 0, 1

class Model(object):
    def __init__(self, char_dim, feat_dim, hidden_dim, char_size, feat_sizes):
        self._char_dim = char_dim
        self._feat_dim = feat_dim

        self._pc = dy.ParameterCollection()

        if config.adam:
            self._trainer = dy.AdamTrainer(self._pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
        else:
            # self._trainer = dy.AdadeltaTrainer(self._pc)
            trainer = dy.SimpleSGDTrainer(self._pc, config.learning_rate)
            trainer.set_clip_threshold(config.clip_threshold)

        # self._trainer.set_clip_threshold(1.0)

        self.params = dict()

        self.lp_c = self._pc.add_lookup_parameters((char_size, char_dim))
        self.lp_feats = []
        for idx in range(len(feat_sizes)):
            self.lp_feats.append(self._pc.add_lookup_parameters((feat_sizes[idx], feat_dim), init=dy.ConstInitializer(0.)))

        # self._pdrop_embs = pdrop_embs
        # self._pdrop_lstm = pdrop_lstm
        # self._pdrop_mlp = pdrop_mlp

        self.LSTM_builders = []

        f = dy.VanillaLSTMBuilder(1, char_dim, hidden_dim, self._pc)
        b = dy.VanillaLSTMBuilder(1, char_dim, hidden_dim, self._pc)

        self.LSTM_builders.append((f, b))
        for i in range(config.layers - 1):
            f = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
            b = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
            self.LSTM_builders.append((f, b))

        self.dec_LSTM = dy.VanillaLSTMBuilder(1, hidden_dim, hidden_dim, self._pc)

        self.MLP = self._pc.add_parameters((char_dim + feat_dim * 6 + 6, hidden_dim))
        self.MLP_bias = self._pc.add_parameters((hidden_dim))
        self.classifier = self._pc.add_parameters((hidden_dim, char_size))
        self.classifier_bias = self._pc.add_parameters((char_size))
        self.MLP_attn = self.add_parameters((char_dim + feat_dim * 6 + 6, hidden_dim))
        self.MLP_attn_bias = self.add_parameters((hidden_dim))
        self.attn_weight = self._pc.add_parameters((char_dim))

    def run(self, triple, isTrain):

        MLP = dy.parameter(self.MLP)
        MLP_bias = dy.parameter(self.MLP_bias)
        MLP_attn = dy.parameter(self.MLP_attn)
        MLP_attn_bias = dy.parameter(self.MLP_attn_bias)
        attn_weight = dy.parameter(self.attn_weight)

        s, t, f = triple
        char_embs = [self.lp_c[c] for c in s]
        top_recur = utils.biLSTM(self.LSTM_builders, char_embs)
        key = dy.concatenate_cols(top_recur)

        feat_embs = []
        for idx in range(len(self.lp_feats)):
            if idx < len(f):
                feat_embs.append(self.lp_feats[idx][f[idx]])
            else:
                feat_embs.append(dy.inputVector(np.zeros(self._feat_dim)))
        feat_embs = dy.concatenate(feat_embs)

        prev_char = BOW
        pred_word = []
        losses = []

        state = self.dec_LSTM.initial_state()
        while prev_char != EOW:
            tmp = dy.concatenate([self.lp_c[prev_char], feat_embs, dy.inputVector(np.binary_repr(idx, 6))])
            query = dy.dot_product(attn_weight, dy.affine_transform([MLP_attn_bias, MLP_attn, tmp]))
            attn_vec = dy.transpose(key) * query
            value = key * attn_vec
            inp = dy.concatenate([value, tmp])
            inp = dy.affine_transform([MLP_bias, MLP, inp])
            top_recur = state.add_input(inp)
            score = dy.affine_transform([self.classifier_bias, self.classifier, top_recur])
            if isTrain:
                losses.append(dy.pickneglogsoftmax(score, t[idx]))
                prev_char = t[idx]
            else:
                pred_char = score.npvalue().argmax()
                pred_word.append(pred_char)
                prev_char = pred_char

        return pred_word, losses


















