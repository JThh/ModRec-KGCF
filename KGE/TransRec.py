# -*- coding: utf-8 -*-
# @Time    : 2021/9/30
# @Author  : Jiatong Han (Modifier), Hui Wang (Original Author)
# @Email   : jiatong.han@u.nus.edu

r"""
TransRec
################################################
Reference:
    Ruining He et al. "Translation-based Recommendation." In RecSys 2017.
"""


# PyTorch imports
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)

# Workspace imports
from MLP.MLP_dataset import SequenceEnrolmentDataset, make_datasets
from MLP.utils import seq_train_one_epoch, seq_test, plot_statistics, BPRLoss, EmbLoss, RegLoss, xavier_normal_initialization

# Python imports
import argparse
from time import time
import numpy as np
import pandas as pd
import pickle

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='module_enrolment',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=50,
                        help='Embedding size.')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help="Regularization for each layer")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--neg_sample', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance while training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


class TransRec(nn.Module):
    r"""
    TransRec is translation-based model for sequential recommendation.
    It assumes that the `prev. item` + `user`  = `next item`.
    We use the Euclidean Distance to calculate the similarity in this implementation.
    """

    def __init__(self, user_num, n_items, embedding_size=64):
        # load dataset info
        super().__init__()
        self.USER_ID = 'user_id'
        self.ITEM_ID = 'item_id'
        self.ITEM_SEQ = self.ITEM_ID + '_list'
        self.ITEM_SEQ_LEN = 'item_length'
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = 'neg_' + self.ITEM_ID
        self.max_seq_length = '50'
        self.n_items = n_items

        # load parameters info
        self.embedding_size = embedding_size

        # load dataset info
        self.n_users = user_num

        # print(user_num)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.bias = nn.Embedding(self.n_items, 1, padding_idx=0)  # Beta popularity bias
        self.T = nn.Parameter(torch.zeros(self.embedding_size))  # average user representation 'global'

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_loss = RegLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.__alias__ = 'TransE'

    def _l2_distance(self, x, y):
        return torch.sqrt(torch.sum(torch.square(x - y), dim=-1, keepdim=True))  # [B 1]

    def gather_last_items(self, item_seq, gather_index):
        """Gathers the last_item at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1)
        # print(gather_index)
        # print(item_seq)
        last_items = item_seq.gather(index=gather_index, dim=1)  # [B 1]
        return last_items.squeeze(-1)  # [B]

    def forward(self, user, item_seq, item_seq_len):
        # the last item at the last position
        last_items = self.gather_last_items(torch.tensor(item_seq), torch.tensor(item_seq_len - 1))  # [B]
        user_emb = self.user_embedding(torch.tensor(user))  # [B H]
        last_items_emb = self.item_embedding(last_items)  # [B H]
        T = self.T.expand_as(user_emb)  # [B H]
        seq_output = user_emb + T + last_items_emb  # [B H]
        return seq_output

    def calculate_loss(self, interaction):
        # print(interaction)
        # exit()
        user = torch.tensor(interaction[self.USER_ID])  # [B]
        item_seq = torch.tensor(interaction[self.ITEM_SEQ])  # [B Len]
        item_seq_len = torch.tensor(interaction[self.ITEM_SEQ_LEN])

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        pos_items = torch.tensor(interaction[self.POS_ITEM_ID])  # [B]
        neg_items = torch.tensor(interaction[self.NEG_ITEM_ID])  # [B] sample 1 negative item

        pos_items_emb = self.item_embedding(pos_items)  # [B H]
        neg_items_emb = self.item_embedding(neg_items)

        pos_bias = self.bias(pos_items)  # [B 1]
        neg_bias = self.bias(neg_items)

        # print(seq_output.size())
        # print(pos_items_emb.size())
        # print(neg_items_emb.size())
        # print(neg_bias.size())
        # exit()

        pos_score = pos_bias - self._l2_distance(seq_output, pos_items_emb)
        neg_score = neg_bias - self._l2_distance(seq_output, neg_items_emb)

        bpr_loss = self.bpr_loss(pos_score, neg_score)
        item_emb_loss = self.emb_loss(self.item_embedding(pos_items).detach())
        user_emb_loss = self.emb_loss(self.user_embedding(user).detach())
        bias_emb_loss = self.emb_loss(self.bias(pos_items).detach())

        reg_loss = self.reg_loss(self.T)
        return bpr_loss + item_emb_loss + user_emb_loss + bias_emb_loss + reg_loss

    def predict(self, interaction):
        # print(interaction)
        # exit()
        user = torch.tensor(interaction[self.USER_ID])  # [B]
        item_seq = torch.tensor(interaction[self.ITEM_SEQ])  # [B Len]
        item_seq_len = torch.tensor(interaction[self.ITEM_SEQ_LEN])
        test_item = torch.tensor(interaction[self.ITEM_ID])

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)  # [B H]
        test_bias = self.bias(test_item)  # [B 1]

        scores = test_bias - self._l2_distance(seq_output, test_item_emb)  # [B 1]
        scores = scores.squeeze(-1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user = torch.tensor(interaction[self.USER_ID])  # [B]
        item_seq = torch.tensor(interaction[self.ITEM_SEQ])  # [B Len]
        item_seq_len = torch.tensor(interaction[self.ITEM_SEQ_LEN])

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        test_items_emb = self.item_embedding.weight  # [item_num H]
        test_items_emb = test_items_emb.repeat(seq_output.size(0), 1, 1)  # [user_num item_num H]

        user_hidden = seq_output.unsqueeze(1).expand_as(test_items_emb)  # [user_num item_num H]
        test_bias = self.bias.weight  # [item_num 1]
        test_bias = test_bias.repeat(user_hidden.size(0), 1, 1)  # [user_num item_num 1]

        scores = test_bias - self._l2_distance(user_hidden, test_items_emb)  # [user_num item_num 1]
        scores = scores.squeeze(-1)  # [B n_items]
        return scores

    def get_alias(self):
        return self.__alias__

def main():

    # Get Params
    args = parse_args()
    path = args.path
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    emb_size = args.emb_size
    lr = args.lr
    neg_sample = args.neg_sample
    weight_decay = args.weight_decay
    verbose = args.verbose

    topK = 10
    print("TransE arguments: %s " % (args))
    # model_out_file = 'Pretrain/%s_TransE_%s_%d.h5' %(args.dataset, args.layers, time())

    # make datasets

    print('==> make datasets <==')
    file_path = path + dataset + '.txt'
    names = ['user', 'item', 'rating', 'timestamps']
    data = pd.read_csv(file_path, header=None, sep='\t', names=names)
    d_train, d_test, d_info = make_datasets(data, 1, 1, 1)
    train_num_usr, test_num_usr, train_num_item, test_num_item, train_items_usr_clicked, test_items_usr_clicked = d_info
    # print("NUMBER OF USERS:",num_usr)
    all_train_items = [i for i in range(train_num_item)]
    all_test_items = [i for i in range(test_num_item)]

    # Define DataIterator

    trainIterator = SequenceEnrolmentDataset('train',d_train, batch_size, neg_sample,
                                 all_train_items, train_items_usr_clicked, shuffle=True)
    testIterator = SequenceEnrolmentDataset('test',d_test, batch_size,  shuffle=False)

    # Build model
    model = TransRec(train_num_usr, train_num_item, emb_size)
    # Transfer the model to GPU, if one is available
    model.to(device)
    if verbose:
        print(model)

    optimizer = torch.optim.SGD(model.parameters(), weight_decay=weight_decay, lr=lr)

    # Record performance
    hr_list = []
    mrr_list = []
    loss_list = []

    # Check Init performance
    hr, mrr = seq_test(model, testIterator, topK, test_num_usr, test_items_usr_clicked)
    hr_list.append(hr)
    mrr_list.append(mrr)
    loss_list.append(1)
    # do the epochs now

    for epoch in range(epochs):
        epoch_loss = seq_train_one_epoch( model, trainIterator, optimizer, epoch, device)

        if epoch % verbose == 0:
            hr, mrr = seq_test(model, testIterator, topK, test_num_usr, test_items_usr_clicked)
            hr_list.append(hr)
            mrr_list.append(mrr)
            loss_list.append(epoch_loss)
            # if hr > best_hr:
            #     best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            #     if args.out > 0:
            #         model.save(model_out_file, overwrite=True)
    print("hr for epochs: ", hr_list)
    print("mrr for epochs: ", mrr_list)
    print("loss for epochs: ", loss_list)
    plot_statistics(hr_list, mrr_list, loss_list,model.get_alias(), "./figs")
    with open("metrics", 'wb') as fp:
        pickle.dump(hr_list, fp)
        pickle.dump(mrr_list, fp)

    best_iter = np.argmax(np.array(hr_list))
    best_hr = hr_list[best_iter]
    best_mrr = mrr_list[best_iter]
    print("End. Best Iteration %d:  HR = %.4f, MRR = %.4f. " %
          (best_iter, best_hr, best_mrr))
    # if args.out > 0:
    #     print("The best MLP model is saved to %s" %(model_out_file))


if __name__ == "__main__":
    print("Device available: {}".format(device))
    main()