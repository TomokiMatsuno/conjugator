import dynet as dy
import numpy as np
import config
import utils

BOW, EOW = 0, 1


class Model(object):
    def __init__(self, char_dim, feat_dim, hidden_dim, char_size, feat_sizes):
        self._global_step = 0

        self._char_dim = char_dim
        self._feat_dim = feat_dim
        self._hidden_dim = hidden_dim

        self._pc = dy.ParameterCollection()

        if config.adam:
            self._trainer = dy.AdamTrainer(self._pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)
        else:
            trainer = dy.SimpleSGDTrainer(self._pc, config.learning_rate)
            trainer.set_clip_threshold(config.clip_threshold)

        self.params = dict()

        self.lp_c = self._pc.add_lookup_parameters((char_size, char_dim))
        self.lp_feats = []
        for idx in range(len(feat_sizes)):
            self.lp_feats.append(self._pc.add_lookup_parameters((feat_sizes[idx], feat_dim), init=dy.ConstInitializer(0.)))

        self._pdrop_embs = config.pdrop_embs
        self._pdrop_lstm = config.pdrop_lstm
        self._pdrop_mlp = config.pdrop_mlp

        self.LSTM_builders = []

        f = dy.VanillaLSTMBuilder(1, char_dim, hidden_dim, self._pc)
        b = dy.VanillaLSTMBuilder(1, char_dim, hidden_dim, self._pc)

        self.LSTM_builders.append((f, b))
        for i in range(config.layers - 1):
            f = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
            b = dy.VanillaLSTMBuilder(1, 2 * hidden_dim, hidden_dim, self._pc)
            self.LSTM_builders.append((f, b))

        self.dec_LSTM = dy.VanillaLSTMBuilder(config.layers, hidden_dim, hidden_dim, self._pc)

        self.MLP = self._pc.add_parameters((hidden_dim, hidden_dim * 3 + char_dim + feat_dim * 10))
        self.MLP_bias = self._pc.add_parameters((hidden_dim))
        self.classifier = self._pc.add_parameters((char_size, hidden_dim))
        self.classifier_bias = self._pc.add_parameters((char_size))
        self.MLP_attn = self._pc.add_parameters((hidden_dim * 2, char_dim + feat_dim * 10 + hidden_dim))
        self.MLP_attn_bias = self._pc.add_parameters((hidden_dim * 2))
        self.attn_weight = self._pc.add_parameters((hidden_dim * 2))

    def run(self, triple, isTrain):

        MLP = dy.parameter(self.MLP)
        MLP_bias = dy.parameter(self.MLP_bias)
        MLP_attn = dy.parameter(self.MLP_attn)
        MLP_attn_bias = dy.parameter(self.MLP_attn_bias)
        attn_weight = dy.parameter(self.attn_weight)
        classifier = dy.parameter(self.classifier)
        classifier_bias = dy.parameter(self.classifier_bias)

        s, t, f = triple
        s = [BOW] + s + [EOW]
        t = [BOW] + t + [EOW]
        char_embs = [self.lp_c[c] for c in s]
        top_recur = utils.biLSTM(self.LSTM_builders, char_embs,
                                 dropout_h=self._pdrop_lstm if isTrain else 0.,
                                 dropout_x=self._pdrop_lstm if isTrain else 0.)
        key = dy.concatenate_cols(top_recur[1:-1])

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

        prev_top_recur = dy.inputVector(np.zeros(self._hidden_dim))
        state = self.dec_LSTM.initial_state()
        idx = 0
        while prev_char != EOW:
            tmp = dy.concatenate([self.lp_c[prev_char], feat_embs, prev_top_recur])

            if isTrain:
                tmp = dy.dropout(tmp, self._pdrop_embs)

            h = dy.affine_transform([MLP_attn_bias, MLP_attn, tmp])
            if isTrain:
                h = dy.dropout(h, self._pdrop_mlp)

            query = dy.cmult(attn_weight, dy.rectify(h))
            attn_vec = dy.softmax(dy.transpose(key) * query)
            value = key * attn_vec
            inp = dy.concatenate([value, tmp])
            inp = dy.affine_transform([MLP_bias, MLP, inp])
            h = state.add_input(inp).output()
            top_recur = dy.rectify(h)
            if isTrain:
                top_recur = dy.dropout(top_recur, self._pdrop_mlp)
            prev_top_recur = h
            score = dy.affine_transform([classifier_bias, classifier, top_recur])
            if isTrain:
                losses.append(dy.pickneglogsoftmax(score, t[idx + 1]))
                prev_char = t[idx + 1]
                idx += 1
            else:
                pred_char = score.npvalue().argmax()
                pred_word.append(pred_char)
                prev_char = pred_char
                if len(pred_word) > 30:
                    break

        return pred_word, losses

    def update_parameters(self):
        self._trainer.learning_rate = config.learning_rate * config.decay ** (self._global_step / config.decay_steps)
        self._trainer.update()



















