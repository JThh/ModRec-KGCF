'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for 
    Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan, modified by Jiatong Han, Harshdeep Gupta
'''
import math
import time
import heapq  # for retrieval topK
import numpy as np
import torch
# from numba import jit, autojit
from MLP_dataset import ModuleEnrolmentDataset, SequenceEnrolmentDataset

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_topk = None


def evaluate_model(model, topK: int, full_dataset = ModuleEnrolmentDataset):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _topk
    _model = model
    _testRatings = full_dataset.testRatings
    _testNegatives = full_dataset.testNegatives
    _topk = topK

    hits, ndcgs = [], []
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx, full_dataset)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def evaluate_seq_model(model, full_dataset, topK: int, all_items, items_usr_clicked):

    pred_list = []
    next_list = []
    user_list = []

    for test_input in full_dataset:
        batch_usr, batch_seq, batch_pos, batch_neg = test_input
        feed_dict = {'user_id': batch_usr, 'item_id':batch_pos,
            'item_id_list': batch_seq, 'neg_item_id': batch_neg, 'item_length': len(batch_seq[0])}
        # print(feed_dict)
        # exit()
        pred = model.full_sort_predict(feed_dict) 
        
        pred_list += pred.tolist()
        next_list += list(batch_pos)
        user_list += list(batch_usr)

    sorted_items, sorted_score = SortItemsbyScore(all_items,pred_list,reverse=True,remove_hist=True
                                            ,usr=user_list,usrclick=items_usr_clicked)

    # print('Next list:',next_list)
    # print('Sorted item:',sorted_items)
    # exit()
    hr = Metric_HR(topK, next_list, sorted_items)
    Mrr = Metric_MRR(next_list, sorted_items)

    return (hr, Mrr)


def eval_one_rating(idx, full_dataset = ModuleEnrolmentDataset):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]

    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')

    feed_dict = {
        'user_id': users,
        'item_id': np.array(items),
    }
    predictions = _model.predict(feed_dict)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]

    # Evaluate top rank list
    ranklist = heapq.nlargest(_topk, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def train_one_epoch(model, data_loader, loss_fn, optimizer, epoch_no, device, verbose = 1):
    'trains the model for one epoch and returns the loss'
    print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for feed_dict in data_loader:
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = feed_dict[key].to(dtype = torch.long, device = device)
        # get the predictions
        prediction = model(feed_dict)
        # print(prediction.shape)
        # get the actual targets
        rating = feed_dict['rating']
        
      
        # convert to float and change dim from [batch_size] to [batch_size,1]
        rating = rating.float().view(prediction.size())  
        loss = loss_fn(prediction, rating)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss

def seq_train_one_epoch(model, data_loader, optimizer, epoch_no, verbose = 1):
    'trains the model for one epoch and returns the loss'
    print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for train_input in data_loader:
        batch_usr, batch_seq, batch_pos, batch_neg = train_input
        feed_dict = {'user_id': batch_usr, 'item_id':batch_pos,
            'item_id_list': batch_seq, 'neg_item_id': batch_neg, 'item_length': len(batch_seq[0])}

        # get the predictions
        # prediction = model(feed_dict['user_id'], feed_dict['item_id_list'], feed_dict['item_length'])
        # print(prediction.shape)
        # get the actual targets
        loss = model.calculate_loss(feed_dict)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss

def test(model, topK, full_dataset = ModuleEnrolmentDataset):
    'Test the HR and NDCG for the model @topK'
    # put the model in eval mode before testing
    if hasattr(model,'eval'):
        # print("Putting the model in eval mode")
        model.eval()
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, full_dataset, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Eval: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time()-t1))
    return hr, ndcg

def seq_test(model, full_dataset, topK, all_items, items_usr_clicked):
    'Test the HR and MRR for the model @topK'
    # put the model in eval mode before testing
    if hasattr(model,'eval'):
        # print("Putting the model in eval mode")
        model.eval()
    t1 = time()
    (hr, mrr) = evaluate_seq_model(model, full_dataset, topK, all_items, items_usr_clicked)
    hr, mrr = np.array(hr).mean(), np.array(mrr).mean()
    print('Eval: HR = %.4f, MRR = %.4f [%.1f s]' % (hr, mrr, time()-t1))
    return hr, mrr
    

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def Metric_PrecN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(preds)

    return sum / count

def Metric_RecallN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(target)
    return sum / count

def cal_PR(target_list,predict_list,k=[1,5,10]):

    display_list = []

    for s in k:
        prec = Metric_PrecN(target_list,predict_list,s)
        recall = Metric_RecallN(target_list,predict_list,s)
        display = "Prec@{}:{:g} Recall@{}:{:g}".format(s,round(prec,4),s,round(recall,4))
        display_list.append(display)

    return ' '.join(display_list)


def Metric_HR(TopN, target_list, predict_list):
    sums = 0
    count = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        top_preds = preds[:TopN]

        for target in target_list[i]:
            if target in top_preds:
                sums+=1
            count +=1

    return float(sums) / count

def Metric_MRR(target_list, predict_list):

    # print('Length:',len(target_list),len(predict_list))
    sums = 0
    count = 0
    for i in range(len(predict_list)):
        for t in predict_list[i]:
            if t in target_list:
                rank = target_list.index(t) + 1
                sums += 1 / rank
            count += 1
    return float(sums) / count

def SortItemsbyScore(item_list, item_score_list, reverse=False,remove_hist=False, usr = None, usrclick = None):

    totals = len(item_score_list)
    result_items = []
    result_score = []
    for i in range(totals):
        u = usr[i]
        u_clicks = usrclick[u]
        item_score = item_score_list[i]
        tuple_list = sorted(list(zip(item_list,item_score)),key=lambda x:x[1],reverse=reverse)

        if remove_hist:
            tmp = []
            for t in tuple_list:
                if t[0] not in u_clicks:
                    tmp.append(t)
            tuple_list = tmp

        x, y = zip(*tuple_list)
        sorted_item = list(x)
        sorted_score = list(y)
        result_items.append(sorted_item)
        result_score.append(sorted_score)

    # print(result_items , result_score)
    return result_items,result_score

