import argparse
import pickle
import time
from utils_sl import build_graph, Data, split_validation
from model_LLM import *
import os
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', help='dataset name: Beauty/ml-1m')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
torch.cuda.set_device(0)

def main():
    train_data = pickle.load(open('../Data/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../Data/' + opt.dataset + '/test.txt', 'rb'))

    train_xlsx_file_path = "Search_data/tra_session_long_emb.xlsx"
    test_xlsx_file_path = "Search_data/tes_session_long_emb.xlsx"

    train_df = pd.read_excel(train_xlsx_file_path)
    test_df = pd.read_excel(test_xlsx_file_path)

    train_text_long = train_df['longterm_emb'].tolist()
    test_text_long = test_df['longterm_emb'].tolist()

    train_xlsx_file_path_2 = "Search_data/tra_session_short_emb.xlsx"
    test_xlsx_file_path_2 = "Search_data/tes_session_short_emb.xlsx"

    train_df = pd.read_excel(train_xlsx_file_path_2)
    test_df = pd.read_excel(test_xlsx_file_path_2)

    train_text_short = train_df['shortterm_emb'].tolist()
    test_text_short = test_df['shortterm_emb'].tolist()

    train_data = Data(train_data, train_text_long, train_text_short, shuffle=True)
    test_data = Data(test_data, test_text_long, test_text_short, shuffle=False)


    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'Beauty':
        n_node = 12102
    elif opt.dataset == 'ml-1m':
        n_node = 3417
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_5, hit_10, hit_20, mrr_5, mrr_10, mrr_20, ndcg_5, ndcg_10, ndcg_20 = train_test(model, train_data,
                                                                                            test_data)

        print('Current Result:')
        print('\tRecall@5:\t%.4f\tMRR@5:\t%.4f\tNDCG@5:\t%.4f' % (hit_5, mrr_5, ndcg_5))
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tNDCG@10:\t%.4f' % (hit_10, mrr_10, ndcg_10))
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tNDCG@20:\t%.4f' % (hit_20, mrr_20, ndcg_20))

        # Update best results
        flag = 0
        if hit_5 >= best_result[0]:
            best_result[0] = hit_5
            best_epoch[0] = epoch
            flag = 1
        if mrr_5 >= best_result[1]:
            best_result[1] = mrr_5
            best_epoch[1] = epoch
            flag = 1
        if hit_10 >= best_result[2]:
            best_result[2] = hit_10
            best_epoch[2] = epoch
            flag = 1
        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
            flag = 1
        if hit_20 >= best_result[4]:
            best_result[4] = hit_20
            best_epoch[4] = epoch
            flag = 1
        if mrr_20 >= best_result[5]:
            best_result[5] = mrr_20
            best_epoch[5] = epoch
            flag = 1
        if ndcg_5 >= best_result[6]:
            best_result[6] = ndcg_5
            best_epoch[6] = epoch
            flag = 1
        if ndcg_10 >= best_result[7]:
            best_result[7] = ndcg_10
            best_epoch[7] = epoch
            flag = 1
        if ndcg_20 >= best_result[8]:
            best_result[8] = ndcg_20
            best_epoch[8] = epoch
            flag = 1

        print('Best Result:')
        print('\tRecall@5:\t%.4f\tMRR@5:\t%.4f\tNDCG@5:\t%.4f\tEpoch:\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[6], best_epoch[0], best_epoch[1], best_epoch[6]))
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tNDCG@10:\t%.4f\tEpoch:\t%d,\t%d,\t%d' % (
            best_result[2], best_result[3], best_result[7], best_epoch[2], best_epoch[3], best_epoch[7]))
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tNDCG@20:\t%.4f\tEpoch:\t%d,\t%d,\t%d' % (
            best_result[4], best_result[5], best_result[8], best_epoch[4], best_epoch[5], best_epoch[8]))

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
