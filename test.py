# import torch
# from torch import nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
#
# # a = torch.randint(1, 10, (10,))
# # lens = [3, 2, 3, 2]
# # asegs = torch.split(a, lens, dim=0)
# # padded_seqs = pad_sequence(asegs, batch_first=True)
# # item_embed = nn.Embedding(20, 5)
# # padded_seq_emb = item_embed(padded_seqs)
# # # print(asegs)
# # # print(pad_sequence(asegs, batch_first=True))
# # packed_seq = pack_padded_sequence(padded_seq_emb, lens, batch_first=True,
# #                                   enforce_sorted=False)
# # # 长度为 0 的需要特殊处理一下
# # # print(packed_seq)
# # # mask = (a>0.8)
# # # print([torch.masked_select()for i in range(0, 128)])
# # gru = nn.GRU(5, 10, 1, bias=False, batch_first=True)
# # print(pad_packed_sequence(gru(packed_seq)[0], batch_first=True))
# # print( gru(packed_seq)[1])#gru(packed_seq)[0], '\n',
#
# # a = torch.rand(4, 5, 10)
# # print(torch.max(a))
# # print(torch.topk(a, 2, dim=-1))
# # a[[1, 2], :] = torch.rand(5)
# # print(a)
#
# ratios = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
# res = []
# cnt = 1
# for r1 in ratios:
#     for r2 in ratios:
#         if r1 + r2 > 1:
#             break
#         for r3 in ratios:
#             if r1 + r2 + r3 > 1:
#                 break
#             for r4 in ratios:
#                 if r1 + r2 + r3 + r4 > 1:
#                     break
#                 for r5 in ratios:
#                     if r1 + r2 + r3 + r4 + r5 > 1:
#                         break
#                     for r6 in ratios:
#                         if r1 + r2 + r3 + r4 + r5 + r6 > 1:
#                             break
#                         for r7 in ratios:
#                             if r1 + r2 + r3 + r4 + r5 + r6 + r7 > 1:
#                                 break
#                             for r8 in ratios:
#                                 if r1+r2+r3+r4+r5+r6+r7+r8 > 1:
#                                     break
#                                 res.append([r1, r2, r3, r4, r5, r6, r7, r8])
#                                 cnt += 1
#
# import  numpy as np
# import time
# best_recall = 0
# idxs = list(map(list, np.random.randint(0, 100, (8, 200))))
# # print(idxs)
#
# tar = set(list(np.random.randint(0, 100, (200, ))))
# # print(tar)
# start_time = time.time()
# for i in range(1):
#     for rat in res:
#         nums = [int(r*200) for r in rat[:-1]]
#         nums.append(200 - sum(nums))
#
#         pred_res = []
#         for i, num in enumerate(nums):
#             pred_res += list(idxs[i][:num])
#
#         recall = len(set(pred_res).intersection(tar))
#         if recall > best_recall:
#             best_recall = recall
# # print(best_recall)
# # print(res)
# print(cnt)
# print(time.time()-start_time)
import torch

# def inbatch_softmax(self, user_embeds, pos_embeds, neg_embes):
#     '''
#     Args:
#         self:
#         user_embeds: batch_size * embed_size
#         pos_embeds: batch_size * embed_size
#         neg_embes: batch_size * embed_size
#
#     Returns: cross entropy loss
#
#     '''
#
#     batch_size = user_embeds.shape[0]
#     scores = torch.matmul(user_embeds, torch.cat([pos_embeds, neg_embes], dim=0).transpose(0, 1)) # batch_size * (2batch_size)
#     labels = torch.arange(batch_size)
#     return self.ce(scores, labels)
#
# def inbatch_negonly_softmax(self, user_embeds, pos_embeds, neg_embeds):
#     '''
#     Args:
#         self:
#         user_embeds: batch_size * embed_size
#         pos_embeds: batch_size * embed_size
#         neg_embes: batch_size * embed_size
#
#     Returns: cross entropy loss
#     '''
#     batch_size = user_embeds.shape[0]
#     pos_scores = torch.sum(user_embeds * pos_embeds, dim=-1)
#     neg_scores = torch.matmul(user_embeds, neg_embeds.transpose(0, 1)) # batch_size * batch_size
#     scores = torch.cat([pos_scores, neg_scores], dim=-1) # batch_size * (1+batch_size)
#     labels = torch.zeros(batch_size)
#     return self.ce(scores, labels)


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
from torch import nn

# item_seq = torch.randint(0, 10, (3, 4))
# mask = item_seq > 5
# lens = torch.sum(mask, dim=-1)
# item_seq = torch.masked_select(item_seq, mask)
# seq_segs = torch.split(item_seq, tuple(lens.numpy()), dim=0)
# item_embedding_layer = nn.Embedding(20, 64)
# padded_seqs = pad_sequence(seq_segs, batch_first=True)
# item_seq_embeds = item_embedding_layer(item_seq)
# print(item_seq_embeds.shape)
#
# print(torch.max(item_seq, padded_seqs))


# import numpy as np
# import torch
#
# state_dict = torch.load('runs/Dev_Item_wechat_CE_maxscore_8interest_test_cosine_211111_1225_gru_selfatt')
# item_embedding = state_dict['item_embedding.weight']
#
# cnt = 0
# item_set = set()
# for iid, item in enumerate(item_embedding.cpu().numpy()):
#     if item[0] < 1e-30:
#         item_set.add(iid)
#         cnt += 1
#
# print(item_embedding.shape, cnt, len(item_set))
# print(item_embedding.cpu().numpy())

import torch
import torch.nn.functional as F


# EPS = 0.00001
#
#
# def IIC(z, zt, C=10):
#     P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
#     P = ((P + P.t()) / 2) / P.sum()
#     P[(P < EPS).data] = EPS
#     Pi = P.sum(dim=1).view(C, 1).expand(C, C)
#     Pj = P.sum(dim=0).view(1, C).expand(C, C)
#     return (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()
#
# for i in range(10):
#     z = torch.softmax(torch.rand(1, 10), dim=-1)
#     zt = torch.softmax(torch.rand(1, 10), dim=-1)
#     print(IIC(z, zt))


def sinkhorn(out, mask):
    '''

    Args:
        out: batch_size * n_seq * n_interest
        mask: batch_size * n_seq

    Returns:

    '''
    epsilon = 1  # hyper parameters
    batch_size, n_seq, n_interest = out.shape
    out = out.reshape(batch_size * n_seq, -1)
    mask = mask.reshape(-1).float()
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    # B = Q.shape[1]  # number of samples to assign
    B = torch.sum(mask)  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    Q = Q * mask.unsqueeze(0)  # 有些元素是 padding 元素，需要置为零
    Q /= torch.sum(Q)
    old_Q = Q.clone()

    err = 1e6
    cnt = 0
    while err > 1e-1 and cnt <= 3:
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        Q = torch.where(torch.isnan(Q), torch.zeros_like(Q), Q)  # 考虑到对于 padding 元素已经全部置为零

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
        Q = torch.where(torch.isnan(Q), torch.zeros_like(Q), Q)  # 考虑到对于 padding 元素已经全部置为零
        err = torch.sum(torch.abs(Q - old_Q))
        old_Q = Q.clone()

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t().reshape(batch_size, n_seq, n_interest)


# todo: 有 padding 元素

out = torch.rand(1024, 100, 8)
mask = torch.LongTensor(
    [[1] * 80 + [0] * 20] * 500 + [[1] * 50 + [0] * 50] * 524
)

print('--------')
q = sinkhorn(out, mask)
# print(q)

out = torch.where(mask.unsqueeze(-1) == 0, torch.ones_like(out) * -1e9, out)
probs = torch.softmax(out, dim=-1)
# print(probs)

print(- torch.sum((q * torch.log(probs) * mask.unsqueeze(-1).float())))
