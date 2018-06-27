class Dict:
    def __init__(self, sents, initial_entries=None):
        self.i2x = {}
        self.x2i = {}
        self.appeared_x2i = {}
        self.appeared_i2x = {}
        self.freezed = False
        self.initial_entries = initial_entries

        if initial_entries is not None:
            for ent in initial_entries:
                self.add_entry(ent)

        for ent in sents:
            self.add_entry(ent)

        self.freeze()

    def add_entry(self, ent):
        if ent not in self.x2i:
            if not self.freezed:
                self.i2x[len(self.i2x)] = ent
                self.x2i[ent] = len(self.x2i)
            else:
                self.x2i[ent] = self.x2i['UNK']

    def add_entries(self, seq=None, minimal_count=0):
        if not self.freezed:
            for elem in seq:
                if self.cnt[elem] >= minimal_count and elem not in self.i2x:
                    self.i2x.append(elem)
            self.words_in_train = set(self.i2x)
        else:
            for ent in seq:
                if ent not in self.x2i:
                    self.x2i[ent] = self.x2i['UNK']


    def freeze(self):
        self.freezed = True

PATH2DATA = '/Users/tomoki/NLP_data/conll2018/task1/all/portuguese-dev'
PATH2TRAIN = ''
PATH2DEV = ''

class Vocab(object):
    def __init__(self, data):

        chars = []
        for s in data:
            chars.extend(list(s[0] + s[1]))

        feats = [[] for _ in range(10)]
        for d in data:
            f = d[2]
            idx = 0
            for e in f:
                feats[idx].append(e)
                idx += 1

        self._feat_dicts = []

        for idx in range(10):
            self._feat_dicts.append(Dict(feats[idx], initial_entries=['UNK']))

        self._char_dict = Dict(chars, initial_entries=['<BOW>', '<EOW>', 'UNK'])


    def add_parsefile(self, data):
        chars = []
        for s in data:
            chars.extend(list(s[0] + s[1]))

        feats = [[] for _ in range(10)]
        for d in data:
            f = d[2]
            idx = 0
            for e in f:
                feats[idx].append(e)
                idx += 1

        for idx in range(10):
            self._feat_dicts[idx].add_entries(feats[idx])

        self._char_dict.add_entries(chars)

