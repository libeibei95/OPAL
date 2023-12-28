import pickle

import faiss
import pandas as pd
import torch
import os
import numpy as np
import argparse

from data_iterator_adapt import *
from model_devcl import *
from model_octopus import *
from tqdm import tqdm
import logging
import sys
from datetime import datetime
from time import time

sys.path.append('..')
from tensorboardX import SummaryWriter
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='Item', help='Sess | Item')
    parser.add_argument('--dataset', type=str, default='takatak', help='wechat | takatak')
    parser.add_argument('--model', type=str, default='DevCL',
                        help='Dev | DevCL | GRU4Rec | SRGNN | STAMP | NextItNet | Caser | BERT4Rec')
    parser.add_argument('--filename', type=str, default='test', help='post filename')
    parser.add_argument('--random_seed', type=int, default=19)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_prob', type=float, default=0)
    parser.add_argument('--n_interest', type=int, default=2)
    parser.add_argument('--n_topk_interest', type=int, default=8)  # 为每个用户最多选择 n_topk_interest个兴趣向量
    parser.add_argument('--loss_type', type=str, default='CE', help='BPR | CE | ..')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--test_epoch', type=int, default=1000)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=2, help='number of gnn steps')
    parser.add_argument('--w_sharp', type=float, default=1, help='to make item-interest distribution sharp')
    parser.add_argument('--w_orth', type=float, default=10, help=' to make system interest orth ')
    parser.add_argument('--w_uniform', type=float, default=1,
                        help='to make items uniformly distribute on global interests')
    parser.add_argument('--w_sk', type=float, default=0,
                        help='sk algorithm to make items uniformly distribute on global interests')
    parser.add_argument('--w1', type=float, default=1, help='w for octopus')
    parser.add_argument('--w2', type=float, default=1, help='w for octopus')
    parser.add_argument('--n_step', type=int, default=10, help='to search the subopt allocation')
    parser.add_argument('--best_ckpt_path', type=str, default='runs/', help='the direction to save ckpt')
    parser.add_argument('--eval_type', type=str, default='maxscore',
                        help='avg | maxscore | his_ratios | learned_ratios | subopt | opt | gt_ratio')
    parser.add_argument('--add_selfatt', type=int, default=1, help='whether to add self attention')
    parser.add_argument('--add_gru', type=int, default=1, help='whether to add gru user interest')
    parser.add_argument('--pre_step', type=int, default=5,
                        help='the number of steps to pretrain the model without interest cl when we add interest cl to the model')
    parser.add_argument('--agg_type', type=str, default='soft',
                        help='hard | soft, the type to aggregate items into psnl interests')
    parser.add_argument('--cuda', type=str, default='0', help='the number of cuda')
    parser.add_argument('--log_dir', type=str, default='log', help='the direction of log')
    parser.add_argument('--two_phase', type=int, default=1, help='whether use')
    parser.add_argument('--interest_type', type=str, default='None',
                        help='None | Extra | Plus | ExtraPlus | Agg | ExtraAgg | ExtraAggPlus | Double | PlusDouble')
    return parser.parse_args()


def eval(model, test_data, config, phase='valid', type='avg', ks=None, ratios=None):
    '''
    Args:
        model:
        test_data:
        config:
        phase:
        type: avg|max_score|his_ratios|learned_ratios|subopt|opt|
    Returns:

    '''
    recall, ndcg = [0] * len(ks), 0
    hit = [0] * len(ks)
    num = 0
    k = config.topk
    n_step = config.n_step

    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.device = 0

    # try:
    # gpu_index = faiss.GpuIndexFlatIP(res, model.embedding_size, flat_config)
    index = faiss.IndexFlatIP(model.embedding_size)
    # print('----------')
    # with torch.no_grad():
    if config.model == 'Dev':
        item_embeds = trans_to_cpu(model.W(model.item_embedding.weight).detach()).numpy()
    elif config.model == 'Octopus':
        item_embeds = trans_to_cpu(model.item_embedding.weight.detach()).numpy()
    elif config.model == 'DevCL':
        item_embeds = trans_to_cpu(F.normalize(model.item_embedding.weight, dim=-1, p=2).detach()).numpy()
    index.add(item_embeds[1:])  # filter out item zero

    model.eval()

    if type == 'subopt':
        assert n_step is not None and ratios is not None

    if phase == 'test':
        total_pmi, max_pmi, min_pmi = [0] * len(ks), [0] * len(ks), [9999999] * len(ks)
        npair = sum([len(v) * (len(v) - 1) / 2 for k, v in test_data.inv_idx.items()])
        pmi_mem = {}
    # full_rec_results = []
    with torch.no_grad():
        for uids, seqs, tars, lens in tqdm(test_data):
            uids = trans_to_cuda(torch.LongTensor(uids))
            seqs = trans_to_cuda(torch.LongTensor(seqs))
            lens = trans_to_cuda(torch.LongTensor(lens))

            # if config.model == 'DevCL':
            #     psnl_interest, interest_mask, full_psnl_interest = model(seqs, lens)
            #     full_base_scores = torch.matmul(full_psnl_interest, model.item_embedding.weight.transpose(0, 1))
            # else:
            psnl_interest, interest_mask = model(seqs, lens)
            # interest_mask = torch.nn.functional.softmax(interest_mask / torch.sum(interest_mask, dim=-1, keepdim=True) / 2) #先归一化再 softmax, 这里可以有个超参数来控制扩大倍数
            batch_size, n_interest, embed_size = psnl_interest.shape
            user_embeds = psnl_interest.reshape(-1, psnl_interest.shape[-1])
            nrecall = int(ks[-1] * 2)
            scores, items = index.search(trans_to_cpu(user_embeds.detach()).numpy(), nrecall)
            # scores = scores.reshape(-1, nrecall)
            # scores[trans_to_cpu((interest_mask < 0.0001).reshape(-1)).numpy()] -= 2e9  # todo: 去掉历史中未出现的兴趣
            his_seqs = trans_to_cpu(seqs).numpy()
            # scores = scores * trans_to_cpu(interest_mask.reshape(-1).unsqueeze(-1).detach()).numpy() # interest importance scores
            # batch_recall, batch_hit = eval_max_score(scores.reshape(batch_size, n_interest, -1),
            #                                          items.reshape(batch_size, n_interest, -1), tars, ks, his_seqs)
            batch_recall, batch_hit = eval_max_score(scores.reshape(batch_size, n_interest, -1),
                                                                  items.reshape(batch_size, n_interest, -1), tars, ks,
                                                                  his_seqs)

            recall = [r + br for r, br in zip(recall, batch_recall)]
            hit = [h + hi for h, hi in zip(hit, batch_hit)]
            num += uids.shape[0]  # 累计样本数
            # full_rec_results.extend(rec_results)
            # if phase == 'test':
            #     pmi, max_pmi, min_pmi = eval_max_score_pmi(scores.reshape(batch_size, n_interest, -1),
            #                                                items.reshape(batch_size, n_interest, -1), test_data.inv_idx,
            #                                                [k // 10 for k in ks], pmi_mem, his_seqs, npair)
            #     total_pmi = [t + p for t, p in zip(total_pmi, pmi)]
            #     max_pmi = [max(m, p) for m, p in zip(max_pmi, max_pmi)]
            #     min_pmi = [min(m, p) for m, p in zip(min_pmi, min_pmi)]
    # pickle.dump(full_rec_results, open('res/{}'.format(config.best)))
    if phase == 'valid':
        if ks is None:
            logging.info('Valid: Recall@{:2d}:\t{:.4f}'.format(k, recall / num))
        else:
            for nbr_k, kk in enumerate(ks):
                logging.info('Valid: Recall@{:2d}:\t{:.4f}'.format(kk, recall[nbr_k] / num))
    else:
        if ks is None:
            logging.info('Test: Recall@{:2d}:\t{:.4f}'.format(k, recall / num))
        else:
            for nbr_k, kk in enumerate(ks):
                logging.info('Test: Recall@{:2d}:\t{:.4f}'.format(kk, recall[nbr_k] / num))
            for nbr_k, kk in enumerate(ks):
                logging.info('Test: Hit@{:2d}:\t{:.4f}'.format(kk, hit[nbr_k] / num))
            # for nbr_k, kk in enumerate(ks):
            #     logging.info('Test: Avg_PMI@{:2d}:\t{:.4f}'.format(kk // 10, total_pmi[nbr_k] / num))
            # for nbr_k, kk in enumerate(ks):
            #     logging.info('Test: MAX_PMI@{:2d}:\t{:.4f}'.format(kk // 10, max_pmi[nbr_k]))
            # for nbr_k, kk in enumerate(ks):
            #     logging.info('Test: MIN_PMI@{:2d}:\t{:.4f}'.format(kk // 10, min_pmi[nbr_k]))
    if not os.path.exists('res'):
        os.mkdir('res')
    # full_rec_results = [[r[:2] for r in rec] for rec in full_rec_results]
    # pickle.dump(full_rec_results, open('res/{}'.format(config.best_ckpt_path.split('/')[-1]), 'wb'))
    model.train()  # reset
    if ks is None:
        return [recall / num]
    else:
        return [r / num for r in recall]


def train(writer, model, train_data, valid_data, test_data, config, ratios=None, type='train'):
    '''
    Args:
        writer:
        model:
        train_data:
        valid_data:
        test_data:
        config:
        ratios:
        type: train|finetune

    Returns:

    '''
    step = 0
    loss_sum = 0
    best_metrics = [0]
    trials = 0

    if not os.path.exists('runs'):
        os.mkdir('runs')

    best_model_path = config.best_ckpt_path

    for uids, seqs, poss, negs, lens in train_data:
        torch.cuda.empty_cache()
        uids = trans_to_cuda(torch.LongTensor(uids))
        seqs = trans_to_cuda(torch.LongTensor(seqs))
        poss = trans_to_cuda(torch.LongTensor(poss))
        negs = trans_to_cuda(torch.LongTensor(negs))
        lens = trans_to_cuda(torch.LongTensor(lens))

        model.optimizer.zero_grad()
        step += 1
        if type == 'finetune' and step > 10 * config.test_epoch:
            break

        loss = model.calculate_loss(seqs, lens, poss, negs)
        loss.backward()
        model.optimizer.step()
        loss_sum += loss.item()
        writer.add_scalar("loss", loss.item(), step)

        # record
        if step % config.test_epoch == 0:
            logging.info('Epoch:{:d}\tloss:{:4f}'.format(step // config.test_epoch, loss_sum / config.test_epoch))
            loss_sum = 0
            metrics = eval(model, valid_data, config, phase='valid', type=config.eval_type, ks=[10, 20, 50],
                           ratios=ratios)
            if metrics[-1] > best_metrics[-1]:
                if type == 'train':
                    torch.save(model.state_dict(), best_model_path)
                elif type == 'finetune':
                    torch.save(model.state_dict(), '{}_finetune'.format(best_model_path))
                best_metrics = metrics
                trials = 0
            else:
                trials += 1
                if trials >= 3:
                    model.has_unique_loss = True
                if trials > config.patience and config.model == 'Octopus':
                    break

                if trials > config.patience and model.agg_type == 'soft' and config.two_phase == 1:
                    torch.save(model.state_dict(), '{}_soft'.format(best_model_path))
                    model.load_state_dict(torch.load(config.best_ckpt_path))
                    eval(model, test_data, config, 'test', config.eval_type, ks=[10, 20, 50], ratios=ratios)
                    model.agg_type = 'hard'
                    best_metrics = [0]
                    trials = 0
                    logging.info("=========Change Loss=============")
                elif trials > config.patience and (config.two_phase == 0 or model.agg_type == 'hard'):
                    break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    config = get_args()
    SEED = config.random_seed
    setup_seed(SEED)
    data_path = '../data/{}_data'.format(config.dataset)

    train_data = TrainData(data_path, config.batch_size, maxlen=config.maxlen)
    valid_data = TestData(data_path, config.batch_size, maxlen=config.maxlen)
    test_data = TestData(data_path, config.batch_size, maxlen=config.maxlen, isvalid=False)
    config.n_item, config.n_user = train_data.n_item, train_data.n_user
    filename = '{}_{}_{}in_{}uni_{}sharp_{}orth_{}_{}_{}'.format(config.model, config.dataset, str(config.n_interest),
                                                                 str(config.w_uniform), str(config.w_sharp),
                                                                 str(config.w_orth),
                                                                 datetime.fromtimestamp(time()).strftime('%y%m%d_%H%M'),
                                                                 config.filename, config.interest_type)
    if config.model == 'Octopus':
        filename = '{}_{}_{}in_{}w1_{}w2_{}_{}'.format(config.model, config.dataset, str(config.n_interest),
                                                       str(config.w1), str(config.w2),
                                                       datetime.fromtimestamp(time()).strftime('%y%m%d_%H%M'),
                                                       config.filename)

    if config.filename == '':
        fileflag = input("Please input the title of the checkpoint: ")
        filename += fileflag
    config.best_ckpt_path += filename
    if not os.path.exists('runs_tensorboard'): os.mkdir('runs_tensorboard')
    writer = SummaryWriter('runs_tensorboard/{}'.format(filename))

    if not os.path.exists(config.log_dir): os.mkdir(config.log_dir)
    if os.path.exists('{}/{}'.format(config.log_dir, filename)): os.remove('{}/{}'.format(config.log_dir, filename))
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        filename='{}/{}.log'.format(config.log_dir, filename),
                        level=logging.INFO)
    print(config)

    if config.model == "Dev":
        model = trans_to_cuda(Dev(config))
    elif config.model == "Octopus":
        model = trans_to_cuda(Octopus(config))
    elif config.model == "DevCL":
        model = trans_to_cuda(DevCL(config))

    if config.eval_type == 'subopt':
        ratios = []
        n_interest = config.n_interest
        seq = [0] * 100  # 100 is larger than n_interest

        def search(rem, rem_interest):
            if rem_interest == 1:
                ratios.append(seq[:n_interest - 1] + [rem])
                return 1
            if rem == 0:
                ratios.append(seq[:n_interest - rem_interest] + [0] * rem_interest)
                return 1
            num = 0
            for i in range(0, rem + 1):
                seq[n_interest - rem_interest] = i
                num += search(rem - i, rem_interest - 1)
            return num

        nums = search(config.n_step, n_interest)
        print('There are {} kinds of allocation to choose from.'.format(nums))
    else:
        ratios = None

    # 先将 agg_type 置为 soft, 训练一段时间后使用 hard fine-tune
    # model.agg_type='hard'
    # model.load_state_dict(torch.load('runs/DevCL_takatak_2in_1uni_1sharp_10orth_220108_1540_rmpredictuniqueloss2_None'))
    eval(model, test_data, config, 'test', config.eval_type, ks=[10, 20, 50], ratios=ratios)
    train(writer, model, train_data, valid_data, test_data, config, ratios=ratios)
    model.load_state_dict(torch.load(config.best_ckpt_path))
    eval(model, test_data, config, 'test', config.eval_type, ks=[10, 20, 50], ratios=ratios)

    # logging.info('------------Finetune with validation data----------------')
    # finetune_data = TrainData(data_path, config.batch_size, maxlen=config.maxlen,
    #                           is_finetune=True)  # 将验证集用于 fine-tune 模型
    # train(writer, model, finetune_data, valid_data, test_data, config, ratios=ratios, type='finetune')
    #
    # logging.info('------------Final Prediction ----------------')
    # # model.load_state_dict(torch.load('runs/DevCL_takatak_8in_1.0uni_1.0sharp_10.0orth_211204_1907_test'))
    # eval(model, test_data, config, 'test', config.eval_type, ks=[10, 20, 50], ratios=ratios)


def recalculate():
    print('begin recalculate')
    config = get_args()
    SEED = config.random_seed
    setup_seed(SEED)
    data_path = '../data/{}_data'.format(config.dataset)
    train_data = TrainData(data_path, config.batch_size, maxlen=config.maxlen)
    valid_data = TestData(data_path, config.batch_size, maxlen=config.maxlen)
    test_data = TestData(data_path, config.batch_size, maxlen=config.maxlen, isvalid=False)
    config.n_item, config.n_user = test_data.n_item, test_data.n_user

    if os.path.exists('{}/{}_hit.log'.format(config.log_dir, config.best_ckpt_path)):
        os.remove('{}/{}_hit.log'.format(config.log_dir, config.best_ckpt_path))
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        filename='{}/{}_soft.log'.format(config.log_dir, config.best_ckpt_path), level=logging.INFO)

    if config.model == "Dev":
        model = trans_to_cuda(Dev(config))
    elif config.model == "Octopus":
        model = trans_to_cuda(Octopus(config))
    elif config.model == "DevCL":
        model = trans_to_cuda(DevCL(config))
    # model.agg_type = 'hard'

    writer = SummaryWriter('runs_tensorboard/{}_hit'.format(config.best_ckpt_path))
    model.load_state_dict(torch.load('runs/{}_soft'.format(config.best_ckpt_path)))
    labels = trans_to_cpu(model.get_all_item_label()).detach().numpy()
    item_embeds = trans_to_cpu(model.item_embedding.weight).detach().numpy()
    print('begin pandas')
    item_feat_pd = pd.DataFrame([list(emb) + [lab] for emb, lab in zip(item_embeds, labels)])
    item_feat_pd.to_csv('takatak_item_feat_soft.csv')

    # train(writer, model, train_data, valid_data, test_data, config, ratios=None)
    # eval(model, test_data, config, 'test', config.eval_type, ks=[10, 20, 50], ratios=None)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    # recalculate()
    # replay()
    print('test')
    '''
    python run.py --dataset wechat --n_interest 2 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_2in_1.0uni_1.0sharp_10.0orth_211205_1800_test'
    python run.py --dataset wechat --n_interest 4 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_4in_1.0uni_1.0sharp_10.0orth_211205_2225_test'
    python run.py --dataset wechat --n_interest 8 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_8in_1.0uni_1.0sharp_10.0orth_211205_1802_test'
    python run.py --dataset wechat --n_interest 16 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_16in_1.0uni_1.0sharp_10.0orth_211205_2034_test'

    python run.py --dataset takatak --n_interest 4 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak' --best_ckpt_path 'DevCL_takatak_4in_1.0uni_1.0sharp_10.0orth_211205_2228_test'
    python run.py --dataset takatak --n_interest 8 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak' --best_ckpt_path 'DevCL_takatak_8in_1.0uni_1.0sharp_10.0orth_211205_2330_test'
    python run.py --dataset takatak --n_interest 16 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak' --best_ckpt_path 'DevCL_takatak_16in_1.0uni_1.0sharp_10.0orth_211205_1801_test'

    python run.py --model 'Octopus' --dataset wechat --n_interest 2 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset wechat --n_interest 4 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset wechat --n_interest 8 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset wechat --n_interest 16 --log_dir 'log_octopus'

    python run.py --model 'Octopus' --dataset takatak --n_interest 2 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset takatak --n_interest 4 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset takatak --n_interest 8 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset takatak --n_interest 16 --log_dir 'log_octopus'
    '''
