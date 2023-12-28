# coding=utf-8
import pickle
import random
import time

import numpy as np
import tqdm


class DataFilter:
    def __init__(self, source):
        print(source)
        self.days = 14 if 'wechat' in source else 29
        self.min_sess_len = 4
        self.pos_seqs = self.read('{}/pos_seqs.txt'.format(source))
        self.renumber()
        self.save('{}/pos_seqs_renbr.txt'.format(source))

    def read(self, file):
        res, users = [], [0]
        with open(file, 'r') as f:
            for line in f:
                data = list(map(int, line.strip().split(',')))
                uid, day = data[0], data[1]
                if day > self.days: continue
                if uid != users[-1]:
                    users.append(uid)
                    res.append([[0]] * self.days)
                res[-1][day - 1] = data[2:]
        return res

    def renumber(self):
        lens = [[1 if len(ses) > 0 and ses[0] != 0 else 0 for ses in seq] for seq in self.pos_seqs]
        self.pos_seqs = [seq for seq, seqlen in zip(self.pos_seqs, lens) if
                         sum(seqlen[:-2]) >= self.min_sess_len]  # 筛选历史记录超过四天的用户
        his_seqs = [seq[:-2] for seq in self.pos_seqs]
        train_item_set = set()
        for seq in his_seqs:
            for se in seq:
                for item in se:
                    train_item_set.add(item)
        train_item_set.add(0)
        print('max(train_item_set):', max(train_item_set))
        print('min(train_item_set):', min(train_item_set))
        print('len(train_item_set):', len(train_item_set))
        iid_nbr_dict = {iid: idx for idx, iid in enumerate(sorted(list(train_item_set)))}
        self.pos_seqs = [[[iid_nbr_dict[iid] for iid in ses if iid in train_item_set] for ses in seq] for seq in
                         self.pos_seqs]

    def save(self, file):
        max_iid = 0
        with open(file, 'w') as f:
            for uid, seq in enumerate(self.pos_seqs):
                for day, ses in enumerate(seq):
                    if len(ses) == 0 or len(ses) == 1 and ses[0] == 0: continue
                    max_iid = max(max_iid, max(ses))
                    f.write(','.join([str(uid + 1), str(day + 1)] + list(map(str, ses))) + '\n')
        print('max_iid', max_iid)


class Data:
    def __init__(self, source):
        self.days = 14 if 'wechat' in source else 10
        # self.days = 14 if 'wechat' in source else 5
        self.max_day = 14 if 'wechat' in source else 29
        self.pos_seqs, self.n_user = self.read('{}/pos_seqs_renbr.txt'.format(source))
        self.n_item = max([max(se) for seq in self.pos_seqs for se in seq]) + 1

    def read(self, file):
        res, users = [], [0]
        with open(file, 'r') as f:
            for line in f:
                data = list(map(int, line.strip().split(',')))
                uid, day = data[0], data[1]
                if day < self.max_day - self.days + 1: continue
                if uid != users[-1]:
                    users.append(uid)
                    res.append([[0]] * self.days)
                res[-1][day - (self.max_day - self.days + 1)] = data[2:]
        return res, len(users[1:])

    def get_posseq(self):
        return [[s for se in seq for s in se] for seq in self.pos_seqs]

    def get_inverted_index(self, seqs):
        '''构建倒排索引表 用于求 PMI '''
        inv_idx = {}
        for sid, seq in enumerate(seqs):
            for item in seq:
                if inv_idx.get(item) is not None:
                    inv_idx[item].add(sid)
                else:
                    inv_idx[item] = set([sid])
        return inv_idx

    def get_pmi(self, seqs):
        '''
        计算 PMI 值的函数
        Args:
            seqs:
        Returns:
        '''
        pmi = {}
        item_set = set([iid for seq in seqs for iid in seq])
        total_pairs = len(item_set) * (len(item_set) - 1) // 2
        print(total_pairs)

        for seq in tqdm.tqdm(seqs):
            for i in seq:
                for j in seq:
                    try:
                        pmi[i][j] += 1
                        pmi[i][0] += 1
                    except KeyError:
                        if pmi.get(i) is None:
                            pmi[i] = {0: 0}
                        if pmi[i].get(j) is None:
                            pmi[i][j] = 0
                        pmi[i][j] += 1
                        pmi[i][0] += 1
                    try:
                        pmi[j][i] += 1
                        pmi[j][0] += 1
                    except:
                        if pmi.get(j) is None:
                            pmi[j] = {0: 0, j: 0}
                        if pmi[j].get(i) is None:
                            pmi[j][i] = 0
                        pmi[j][i] += 1
                        pmi[j][0] += 1

        pos_pmi = {}
        cnt = 0
        for i in tqdm.tqdm(pmi):
            for j in pmi[i]:
                cnt += 1
                if j == 0: continue
                pmi[i][j] = np.log(pmi[i][j] * total_pairs / pmi[i][0] / pmi[j][0])
                if pmi[i][j] > 0:
                    if pos_pmi.get(i) is None:
                        pos_pmi[i] = {}
                    if pos_pmi.get(j) is None:
                        pos_pmi[j] = {}
                    pos_pmi[i][j] = pos_pmi[j][i] = pmi[i][j]
        print(cnt)
        return pmi, pos_pmi

    # def get_pmi(self, seqs):
    #     '''
    #     计算 PMI 值的函数
    #     Args:
    #         seqs:
    #     Returns:
    #     '''
    #     pmi = {}
    #     item_set = set([iid for seq in seqs for iid in seq])
    #     total_pairs = len(item_set) * (len(item_set) - 1) // 2
    #     print(total_pairs)
    #
    #
    #     for seq in tqdm.tqdm(seqs):
    #         for i in seq:
    #             for j in seq:
    #                 if pmi.get(i) is None:
    #                     pmi[i] = {0: 0}
    #                 if pmi[i].get(j) is None:
    #                     pmi[i][j] = 0
    #                 pmi[i][j] += 1
    #                 pmi[i][0] += 1
    #                 if pmi.get(j) is None:
    #                     pmi[j] = {0: 0, j: 0}
    #                 if pmi[j].get(i) is None:
    #                     pmi[j][i] = 0
    #                 pmi[j][i] += 1
    #                 pmi[j][j] += 1
    #                 pmi[j][0] += 1
    #
    #     pos_pmi = {}
    #     for i in pmi:
    #         for j in pmi[i]:
    #             if j==0: continue
    #             pmi[i][j] = np.log(pmi[i][j] * total_pairs / pmi[i][0] / pmi[j][0])
    #             if pmi[i][j] > 0:
    #                 if pos_pmi.get(i) is None:
    #                     pos_pmi[i] = {}
    #                 if pos_pmi.get(j) is None:
    #                     pos_pmi[j] = {}
    #                 pos_pmi[i][j] = pos_pmi[j][i] = pmi[i][j]
    #     return pmi, pos_pmi


class TrainData(Data):
    def __init__(self, source, batch_size, maxlen=100, is_finetune=False):
        super(TrainData, self).__init__(source)
        self.batch_size = batch_size
        self.num_seq = len(self.pos_seqs)
        self.maxlen = maxlen
        if is_finetune:
            self.pos_seqs = [[ses for ses in seq[:-1] if len(ses) > 0 and ses[0] != 0] for seq in self.pos_seqs]
        else:
            self.pos_seqs = [[ses for ses in seq[:-2] if len(ses) > 0 and ses[0] != 0] for seq in self.pos_seqs]

        self.reformat_posseqs()
        self.num_seq = len(self.pos_seqs)
        print(len(self.pos_seqs))

    def reformat_posseqs(self):
        self.pos_seqs = [[s for se in seq for s in se] for seq in self.pos_seqs]
        self.pos_seqs = [seq for seq in self.pos_seqs if len(seq) >= 4]  # 约束长度大于等于 3

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def next_item_batch(self):
        idxs = np.random.choice(np.arange(0, self.num_seq), size=self.batch_size, replace=False)
        uids = list(map(lambda idx: idx + 1, idxs))
        seqs, poss, negs = [], [], []
        for idx in idxs:
            le = len(self.pos_seqs[idx])
            tar_idx = np.random.choice(np.arange(3, le))
            seqs.append(self.pos_seqs[idx][:tar_idx])
            tar = self.pos_seqs[idx][tar_idx]
            neg = np.random.randint(1, self.n_item)
            while neg == tar:
                neg = np.random.randint(1, self.n_item)
            poss.append(tar)
            negs.append(neg)

        lens = [len(seq) for seq in seqs]
        max_seqlen = min(max(lens), self.maxlen)
        seqs = [seq + [0] * (max_seqlen - len(seq)) if len(seq) <= max_seqlen else seq[len(seq) - max_seqlen:] for seq
                in seqs]
        lens = [min(le, max_seqlen) for le in lens]
        return uids, seqs, poss, negs, lens

    def __next__(self):
        # return self.next_item_batch() # to test next-click item style
        idxs = np.random.choice(np.arange(0, self.num_seq), size=self.batch_size, replace=False)
        uids = list(map(lambda idx: idx + 1, idxs))
        seqs, poss, negs = [], [], []
        for idx in idxs:
            le = len(self.pos_seqs[idx])
            tar_idx = np.random.choice(np.arange(3, le))
            # tar_idx = np.random.choice(np.arange(3, le))
            seqs.append(self.pos_seqs[idx][:tar_idx])
            pos_list = list(self.pos_seqs[idx][tar_idx:])
            pos_set = set(pos_list)

            # tar = self.pos_seqs[idx][tar_idx]
            tar = random.choice(pos_list)
            neg = np.random.randint(1, self.n_item)
            while neg in pos_set:
                neg = np.random.randint(1, self.n_item)
            poss.append(tar)
            negs.append(neg)

        lens = [len(seq) for seq in seqs]
        max_seqlen = min(max(lens), self.maxlen)
        seqs = [seq + [0] * (max_seqlen - len(seq)) if len(seq) <= max_seqlen else seq[len(seq) - max_seqlen:] for seq
                in seqs]
        lens = [min(le, max_seqlen) for le in lens]
        return uids, seqs, poss, negs, lens


class TestData(Data):
    def __init__(self, source, batch_size, isvalid=True, maxlen=100):
        super(TestData, self).__init__(source)
        self.inv_idx = self.get_inverted_index(self.get_posseq())
        self.batch_size = batch_size
        self.num_seq = len(self.pos_seqs)
        self.curr_batch = 0
        self.taridx = -2 if isvalid else -1
        self.maxlen = maxlen

        # 筛选 taridx 有交互的用户，并把非空序列拼接在一块儿。
        self.pos_seqs = [[ses for ses in seq[:self.taridx] if len(ses) > 0 and ses[0] != 0] + [seq[self.taridx]] for seq
                         in self.pos_seqs
                         if len(seq[self.taridx]) > 0 and seq[self.taridx][0] != 0]
        self.pos_seqs = [seq for seq in self.pos_seqs if len(seq) > 1]
        self.num_seq = len(self.pos_seqs)
        self.nbatch = self.num_seq // self.batch_size
        if self.num_seq % self.batch_size:
            self.nbatch += 1
        print(len(self.pos_seqs))

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        idxs = np.arange(self.curr_batch * self.batch_size, min(self.num_seq, (self.curr_batch + 1) * self.batch_size))
        uids = list(map(lambda idx: idx + 1, idxs))
        if len(idxs) == 0:
            return
        seqs, tars = [], []
        for idx in idxs:
            his_seq = [item for ses in self.pos_seqs[idx][:-1] for item in ses]
            gts = self.pos_seqs[idx][-1]
            seqs.append(his_seq)
            tars.append(gts)

        lens = list(map(len, seqs))
        max_seqlen = min(max(lens), self.maxlen)
        seqs = [seq + [0] * (max_seqlen - len(seq)) if len(seq) <= max_seqlen else seq[len(seq) - max_seqlen:] for seq
                in seqs]
        lens = [min(le, max_seqlen) for le in lens]
        self.curr_batch += 1
        if self.curr_batch >= self.nbatch:
            self.curr_batch = 0
            raise StopIteration
        return uids, seqs, tars, lens


if __name__ == '__main__':
    # train_data = TrainData('../data/takatak_data', 16)
    # valid_data = TestData('../data/takatak_data', 16)
    test_data = TestData('../data/takatak_data', 16, isvalid=False)
    # seqs = []
    # for te in test_data:
    #     seqs.extend(te[1])
    # pickle.dump(seqs, open('seqs.pkl', 'wb'))
    seqs = [[item for ses in seq[:-1] for item in ses] for seq in test_data.pos_seqs]
    pospmi, pmi = test_data.get_pmi(seqs)
    pickle.dump(pospmi, open('takatak_pospmi_5.pkl', 'wb'))
    pickle.dump(pmi, open('takatak_pmi_5.pkl', 'wb'))
    # print(pmi)
    # t1 = time.time()
    # cnt = 0
    # for tr in train_data:
    #     cnt += 1
    #     # print(cnt)
    #     if cnt > 10240:
    #         break
    # print(time.time() - t1)
