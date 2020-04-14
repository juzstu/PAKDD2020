#!/usr/bin/env python
# -*- coding:utf-8 -*-

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')


def build_model(train_, feats, label, id_col, cart_cols, use_cart=True, is_shuffle=True):
    print('Use {} features ...'.format(len(feats)))
    n_splits = 5
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 8,
        'verbose': -1,
    }
    train_ids = train_[id_col].unique()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_ids), start=1):
        print(f'the {n_fold} training start ...')
        for_train = train_.loc[train_[id_col].isin(train_ids[train_idx])]
        for_valid = train_.loc[train_[id_col].isin(train_ids[valid_idx])]
        print(f'for train {id_col}:{len(train_idx)}\nfor valid {id_col}:{len(valid_idx)}')
        if use_cart:
            dtrain = lgb.Dataset(for_train[feats], label=for_train[label], categorical_feature=cart_cols)
            dvalid = lgb.Dataset(for_valid[feats], label=for_valid[label], categorical_feature=cart_cols)
        else:
            dtrain = lgb.Dataset(for_train[feats], label=for_train[label])
            dvalid = lgb.Dataset(for_valid[feats], label=for_valid[label])

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        clf.save_model(f'pakdd_lgb_{n_fold}.txt')


if __name__ == "__main__":
    data_path = '/tcdata'

    cate_cols = ['smart_3_normalized', 'smart_199raw']
    train = pd.read_csv(f'{data_path}/more_for_round2_train.csv')
    use_cols = [c for c in train.columns if c not in ['serial_number', 'dt', 'label', 'fault_time']]
    build_model(train, use_cols, 'label', 'serial_number', cate_cols)
