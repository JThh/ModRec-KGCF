# Modified by Jiatong Han
# Author: Harshdeep Gupta
# Date: 22 November, 2018
# Description: A file for implementing the Dataset interface of PyTorch

import scipy.sparse as sp
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
np.random.seed(7)


def make_datasets(data, len_Seq, len_Tag, len_Pred):

    #file_path = 'input/u.data'

    p = data.groupby('item')['user'].count().reset_index().rename(
        columns={'user': 'item_count'})
    data = pd.merge(data, p, how='left', on='item')
    data = data[data['item_count'] > 5].drop(['item_count'], axis=1)

    # ReMap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1, len(item_unique) + 1)))
    item_map[-1] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[-1] = 0
    all_user_count = len(item_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    # Get user session
    data = data.sort_values(by=['user', 'timestamps']).reset_index(drop=True)

    # 生成用户序列
    user_sessions = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    train_items_usr_clicked = {}
    test_items_usr_clicked = {}

    for index, row in user_sessions.iterrows():
        user = row['user']
        items = row['item_list']

        test_item = items[-1*len_Pred:]
        test_seq = items[-1 * (len_Pred + len_Seq):-1*len_Pred]
        test_users.append(user)
        test_seqs.append(test_seq)
        test_targets.append(test_item)

        train_build_items = items[:-1*len_Pred]

        train_items_usr_clicked[user] = train_build_items
        # test_items_usr_clicked[user] = items[:-1*len_Pred:]

        for i in range(len_Seq, len(train_build_items) - len_Tag + 1):
            item = train_build_items[i:i + len_Tag]
            seq = train_build_items[max(0, i - len_Seq):i]

            train_users.append(user)
            train_seqs.append(seq)
            train_targets.append(item)

    d_train = pd.DataFrame(
        {'user': train_users, 'seq': train_seqs, 'target': train_targets})

    d_test = pd.DataFrame(
        {'user': test_users, 'seq': test_seqs, 'target': test_targets})

    d_info = (len(train_users), len(test_users), len(train_targets), len(test_targets),
              train_items_usr_clicked, test_items_usr_clicked)

    return d_train, d_test, d_info


class SequenceEnrolmentDataset(Dataset):
    def __init__(self,
                 mode,
                 data,
                 batch_size=128,
                 neg_sample=1,
                 all_items=None,
                 items_usr_clicked=None,
                 shuffle=True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx = 0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums = self.datasize - self.idx

        cur = self.data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u in cur['user']:
                user_item_set = set(self.all_items) - \
                    set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set, self.neg_count))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)


class ModuleEnrolmentDataset(Dataset):
    'Characterizes the dataset for PyTorch, and feeds the (user,item) pairs for training'

    def __init__(self, file_name, num_negatives_train=1, num_negatives_test=1):
        'Load the datasets from disk, and store them in appropriate structures'

        self.trainMatrix = self.load_rating_file_as_matrix(
            file_name + ".train")
        self.num_users, self.num_items = self.trainMatrix.shape
        # make training set with negative sampling
        self.user_input, self.item_input, self.ratings = self.get_train_instances(
            self.trainMatrix, num_negatives_train)
        # make testing set with negative sampling
        self.testRatings = self.load_rating_file_as_list(
            file_name + ".test")
        self.testNegatives = self.create_negative_file(
            num_samples=num_negatives_test)

        assert len(self.testRatings) == len(self.testNegatives)

    def __len__(self):
        'Denotes the total number of rating in test set'
        return len(self.user_input)

    def __getitem__(self, index):
        'Generates one sample of data'

        # get the train data
        user_id = self.user_input[index]
        item_id = self.item_input[index]
        rating = self.ratings[index]

        return {'user_id': user_id,
                'item_id': item_id,
                'rating': rating}

    def get_train_instances(self, train, num_negatives):
        user_input, item_input = [], []
        num_users, num_items = train.shape
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(1, num_items)
                # while train.has_key((u, j)):
                while (u, j) in train:
                    j = np.random.randint(1, num_items)
                user_input.append(u)
                item_input.append(j)
        return user_input, item_input

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def create_negative_file(self, num_samples=100):
        negativeList = []
        for user_item_pair in self.testRatings:
            user = user_item_pair[0]
            item = user_item_pair[1]
            negatives = []
            for t in range(num_samples):
                j = np.random.randint(1, self.num_items)
                while (user, j) in self.trainMatrix or j == item:
                    j = np.random.randint(1, self.num_items)
                negatives.append(j)
            negativeList.append(negatives)
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()
        return mat

