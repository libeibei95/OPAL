import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
import torch.nn.functional as F
from loss import *
from utils import *
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class AttLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttLayer, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, qvec, kmat):
        '''
        Args:
            qvec: batch_size * hidden_dim
            kmat: batch_size * n_interest * hidden_dim

        Returns:
        '''
        # scores = self.linear(qvec).unsqueeze(1).matmul(kmat.transpose(1, 2)) # bs*ni
        scores = qvec.unsqueeze(1).matmul(kmat.transpose(1, 2)) / 0.1 # bs*ni
        probs = torch.softmax(scores, dim=-1)
        interest = torch.matmul(probs, kmat).squeeze() # bs*embed_size
        return interest


class DevCL(nn.Module):

    def __init__(self, config):
        super(DevCL, self).__init__()

        # load parameters info
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.loss_type = config.loss_type
        self.num_layers = config.num_layers
        self.dropout_prob = config.dropout_prob
        self.add_selfatt = config.add_selfatt > 0
        self.add_gru = config.add_gru > 0
        self.agg_type = config.agg_type
        self.n_items = config.n_item
        self.temp = config.temp
        self.w_uniform = config.w_uniform  # 约束在全局交互数据在各个兴趣向量上分布均匀
        self.w_orth = config.w_orth  # 约束全局兴趣向量比较正交
        self.w_sharp = config.w_sharp  # 约束item属于一个全局兴趣向量
        self.w_sk = config.w_sk
        self.interest_type = config.interest_type
        self.has_unique_loss = True

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.n_interest = config.n_interest
        self.W = nn.Linear(self.embedding_size, self.embedding_size)
        self.selfatt_W = nn.Linear(self.n_interest, self.n_interest, bias=False)
        self.interest_embedding = nn.Embedding(self.n_interest, self.embedding_size)  # 设定为八个兴趣方向

        # GRU
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            nn.ReLU()
        )

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        if 'Agg' in self.interest_type:
            self.interest_agg_layer = AttLayer(self.hidden_size)
        if 'Double' in self.interest_type:
            self.interest_agg_layer = AttLayer(self.hidden_size)
            self.interest_gate = nn.Sequential(
                nn.Linear(self.hidden_size*5, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            )

        self.init_parameter()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # 参数的正则项系数
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

    def init_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name == 'interest_embedding':
                torch.nn.init.orthogonal_(weight.data)
                print(name)
            else:
                weight.data.uniform_(-stdv, stdv)

    def get_orth_loss(self, x):
        '''
        Args:
            x: batch_size * embed_size; Orthogonal embeddings
        Returns:
        '''
        num, embed_size = x.shape
        sim = x.reshape(-1, embed_size).matmul(x.reshape(-1, embed_size).transpose(0, 1))
        diff = sim - trans_to_cuda(torch.eye(sim.shape[1]))
        regloss = diff.pow(2).sum() / (num * num)
        return regloss

    def forward(self, item_seq, item_seq_len, istrain=False):
        with torch.no_grad():
            w = self.item_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.item_embedding.weight.copy_(w)
            w = self.interest_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_embedding.weight.copy_(w)

        batch_size, n_seq = item_seq.shape
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)

        psnl_interest = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1,
                                                                           1)  # bs * n_interest * embed_size
        interest_cl = self.w_orth * self.get_orth_loss(self.interest_embedding.weight)

        for i in range(1):  # 迭代次数可以变成超参数
            scores = item_seq_emb.matmul(psnl_interest.transpose(1, 2)) / self.temp
            scores = scores.reshape(batch_size * n_seq, -1)
            mask = (item_seq > 0).reshape(-1)

            probs = torch.softmax(scores.reshape(batch_size, n_seq, -1), dim=-1) * (item_seq > 0).float().unsqueeze(-1)

            if self.w_uniform:
                interest_prb_vec = torch.sum(probs.reshape(batch_size * n_seq, -1), dim=0) / torch.sum(
                    mask)  # n_interest 1-dim vector
                # print(probs.shape, interest_prb_vec.shape)
                interest_cl += self.w_uniform * interest_prb_vec.std() / interest_prb_vec.mean()
                #todo: 求和均匀向量的交叉熵

            psnl_interest = probs.transpose(1, 2).matmul(item_seq_emb)
            psnl_interest = F.normalize(psnl_interest, dim=-1, p=2)

            sys_interest_vec = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            interest_mask = torch.sum(probs, dim=1)  # batch_size * n_interest
            psnl_interest = torch.where(interest_mask.unsqueeze(-1) > 0, psnl_interest, sys_interest_vec) #todo: 这里可以设置一个阈值
            batch_size, seq_len, n_interest = probs.shape

            if self.agg_type == 'hard' and self.add_gru:
                probs_maxval = torch.max(probs, dim=-1)[0]
                probs_final = torch.where(probs < probs_maxval.unsqueeze(-1) - 1e-9, torch.zeros_like(probs), probs)
                # 设置 unique loss: 输入 softmax 的应该是分数而不是概率
                interest_cl += self.w_sharp * F.cross_entropy(scores[mask, :],
                                                              torch.argmax(scores[mask, :], dim=-1))  # 分配到最大兴趣的对比损失

                # 必须是基于概率计算的，不然更新不了参数
                if self.w_uniform:
                    interest_prb_vec = torch.sum(probs_final, dim=0) / probs.shape[0]  # n_interest 1-dim vector
                    interest_cl += self.w_uniform * interest_prb_vec.std() / interest_prb_vec.mean()

                # probs_final = torch.where(probs_final > 1 / self.n_interest / 4, probs_final, torch.zeros_like(probs_final))
                probs_final = probs_final.transpose(1, 2).reshape(-1, n_seq)

                mask = probs_final > 0
                lens = torch.sum(mask, dim=-1)

                gru_item_seq = item_seq.unsqueeze(1).repeat(1, n_interest, 1).reshape(-1)[mask.reshape(-1)]
                probs_final = probs_final.reshape(-1)[mask.reshape(-1)]
                gru_item_seq_segs = torch.split(gru_item_seq, tuple(lens.cpu().numpy()), dim=0)
                probs_segs = torch.split(probs_final, tuple(lens.cpu().numpy()), dim=0)
                padded_seqs = pad_sequence(gru_item_seq_segs, batch_first=True)
                padded_probs = pad_sequence(probs_segs, batch_first=True)
                # 在输入 GRU 之前乘上原来的注意力系数, 不然没法对全局类别向量进行反向传播, 去掉 soft 后只有 hard 兴趣性能变差，可能是因为没有对全局类别向量反向传播
                padded_seq_embeds = self.item_embedding(padded_seqs) #* padded_probs.unsqueeze(-1) # 测试一下去掉概率的版本，即依靠 soft 部分对全局类别向量进行更新
                pos_lens = torch.max(torch.ones_like(lens), lens)
                packed_seq = pack_padded_sequence(padded_seq_embeds, tuple(pos_lens.cpu().numpy()), batch_first=True,
                                                  enforce_sorted=False)
                packed_output, hidden = self.gru_layers(packed_seq)
                hidden = self.mlp(hidden[0])  # (bs*n_interest, embed_size)
                hidden = F.normalize(hidden, dim=-1, p=2)

                base_interest = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1,
                                                                                                             self.embedding_size)
                seq_interest = torch.where((lens == 0).unsqueeze(-1), base_interest, hidden).reshape(batch_size,
                                                                                                     n_interest,
                                                                                                     self.embedding_size)
                #todo: 调整两者比例
                psnl_interest = 0.5 * psnl_interest + 0.5 * seq_interest
                psnl_interest = F.normalize(psnl_interest, dim=-1, p=2)

        # add global psnl embedding with GRU，用户对物品的偏好分数 = 某个单独的兴趣对物品的偏好分数 + 全局个性化偏好对物品的偏好分数
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.mlp(gru_output)
        full_psnl_emb = F.normalize(self.gather_indexes(gru_output, item_seq_len - 1), p=2, dim=-1)

        # 计算用户整体兴趣向量与各个兴趣点之间的相关性 interest importance scores
        imp_probs = torch.softmax(full_psnl_emb.unsqueeze(1).matmul(psnl_interest.transpose(1, 2)).squeeze() / self.temp, dim=-1)
        interest_mask = imp_probs # 将 interest_mask 用于表示各个兴趣向量的重要程度

        if 'Agg' in self.interest_type:
            psnl_interest = self.interest_agg_layer(full_psnl_emb, psnl_interest).unsqueeze(1)
        if 'Double' in self.interest_type:
            # calculate gate scores
            # agg_interest =  self.interest_agg_layer(full_psnl_emb, psnl_interest).unsqueeze(1)
            # gate_input = torch.cat([agg_interest.repeat(1, self.n_interest, 1), psnl_interest, agg_interest-psnl_interest, agg_interest+psnl_interest, agg_interest*psnl_interest], dim=-1)
            # psnl_interest = agg_interest  + self.interest_gate(gate_input) * psnl_interest # 加上通过 attention 聚合的多兴趣
            psnl_interest = self.interest_agg_layer(full_psnl_emb, psnl_interest).unsqueeze(
                1) + psnl_interest  # 加上通过 attention 聚合的多兴趣

        '''开始计算最终的用户嵌入'''
        if istrain:
            if 'Plus' in self.interest_type:
                psnl_interest = F.normalize(psnl_interest + full_psnl_emb.unsqueeze(1), p=2, dim=-1)
            return psnl_interest, interest_cl, interest_mask, full_psnl_emb
        else:
            if 'Plus' in self.interest_type:
                psnl_interest = F.normalize(psnl_interest + full_psnl_emb.unsqueeze(1), p=2, dim=-1)
            if 'Extra' in self.interest_type:
                psnl_interest = torch.cat([psnl_interest, full_psnl_emb.unsqueeze(1)], dim=1)
            return psnl_interest, interest_mask

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


    def get_all_item_label(self):
        ''' get hard label of each item'''
        with torch.no_grad():
            w = self.item_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.item_embedding.weight.copy_(w)
            w = self.interest_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_embedding.weight.copy_(w)

        item_seq_emb = self.item_embedding.weight  # n_item * embed_size
        interest_emb = self.interest_embedding.weight  # n_interest
        scores = torch.matmul(item_seq_emb, interest_emb.transpose(0, 1))  # n_item * n_interest
        labels = torch.argmax(scores, dim=-1)
        return labels

    def calculate_loss(self, item_seq, item_seq_len, pos_items, neg_items):
        psnl_user_embeds, interest_reg, user_embed_mask, full_user_embed = self.forward(item_seq, item_seq_len,
                                                                       istrain=True)
        batch_size, n_interest, embed_size = psnl_user_embeds.shape

        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)

        if self.loss_type == 'BPR':
            scores = torch.sum(psnl_user_embeds * pos_items_emb.unsqueeze(1), dim=-1)
            interest_idx = torch.argmax(scores, dim=1)
            user_embeds = psnl_user_embeds.reshape(-1, embed_size)[
                          trans_to_cuda(torch.arange(0, batch_size)) * n_interest + interest_idx, :
                          ]
            pos_score = torch.sum(user_embeds * pos_items_emb, dim=-1)
            neg_score = torch.sum(user_embeds * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + interest_reg
        else:
            pos_scores = torch.sum(psnl_user_embeds * pos_items_emb.unsqueeze(1), dim=-1)
            neg_scores = psnl_user_embeds.reshape(-1, embed_size).matmul(neg_items_emb.transpose(0, 1)).reshape(
                batch_size, -1, batch_size)
            scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)

            # 乘上兴趣向量的重要系数 interest importance scores
            # scores = scores * user_embed_mask.unsqueeze(-1)
            # scores = torch.max(scores, dim=1)[0]
            # loss = self.loss_fct(scores / self.temp, trans_to_cuda(torch.zeros(batch_size).long()))

            if 'Extra' in self.interest_type:
                pos_scores = torch.sum(full_user_embed * pos_items_emb, dim=-1)
                neg_scores = full_user_embed.matmul(neg_items_emb.transpose(0, 1))
                scores_full = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
                scores = torch.cat([scores, scores_full.unsqueeze(1)], dim=1)

            # todo: 约束在推荐的时候，每个候选 item 尽可能只与用户的一个兴趣相关
            # scores: batch_size * n_interest * n_item
            # if self.has_unique_loss:
            # unique_scores = scores.transpose(1, 2).reshape(-1, scores.shape[1])
            # interest_reg += 0.01 * F.cross_entropy(unique_scores / self.temp, torch.argmax(unique_scores, dim=-1))
            # interest_reg += F.cross_entropy(unique_scores / self.temp, torch.argmax(scores, dim=-1))
            scores = torch.max(scores, dim=1)[0]
            loss = self.loss_fct(scores / self.temp,  trans_to_cuda(torch.zeros(batch_size).long()))
            return interest_reg + loss
