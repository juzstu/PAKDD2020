#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import entropy
import datetime

warnings.filterwarnings('ignore')


def fill_miss_date(df, id_col):
    date_df = df.groupby(id_col)['dt'].agg(['min', 'max']).reset_index()
    date_df = date_df[[id_col, 'min']].append(pd.DataFrame(date_df[[id_col, 'max']].values, columns=[id_col, 'min']))
    date_df.columns = [id_col, 'dt']
    date_df = date_df.set_index('dt')
    date_df['tmp'] = 1
    date_df = date_df.groupby(id_col).resample('1D', closed='left')['tmp'].count().reset_index()
    del date_df['tmp']
    return date_df


def get_pos_df_from_tag(ids, cols, start_date, end_date):
    tmp_date = pd.date_range(start_date, end=end_date, freq='M')
    tmp_date = [''.join(str(t).split('-')[:2]) for t in tmp_date]
    tmp = pd.DataFrame()
    for i in tmp_date:
        print(f'use date of {i} ...')
        tmp_log = pd.read_csv(f'{data_path}/disk_sample_smart_log_{i}.csv', usecols=cols)
        tmp_log['serial_number'] = tmp_log['model'].astype(str) + '_' + tmp_log['serial_number']
        tmp_log = tmp_log[tmp_log['serial_number'].isin(ids)]
        tmp = tmp.append(tmp_log)
    return tmp


def get_sample_df_for_negative(ids, frac, cols, start_date, end_date):
    tmp_date = pd.date_range(start_date, end=end_date, freq='M')
    tmp_date = [''.join(str(t).split('-')[:2]) for t in tmp_date]
    tmp_df = pd.DataFrame()
    for i in tmp_date:
        tmp_log = pd.read_csv(f'{data_path}/disk_sample_smart_log_{i}.csv', usecols=cols)
        tmp_log['serial_number'] = tmp_log['model'].astype(str) + '_' + tmp_log['serial_number']
        tmp_log = tmp_log[~tmp_log['serial_number'].isin(ids)]
        print(f'date of {i} has {tmp_log.shape[0]} rows')
        sample_ids = [i for i in pd.Series(tmp_log['serial_number'].unique()).sample(frac=frac, random_state=1).values]
        tmp_log = tmp_log[tmp_log['serial_number'].isin(sample_ids)]
        print(f'after sampling, now has {tmp_log.shape[0]} rows')
        tmp_df = tmp_df.append(tmp_log)
        print('#'*100)
    return tmp_df


def calc_entropy(arr):
    return entropy(arr, base=2)


def get_rolling_mean(grp, freq):
    return grp.rolling(freq).mean()


def get_rolling_std(grp, freq):
    return grp.rolling(freq).std()


def get_rolling_ent(grp, freq):
    return grp.rolling(freq).apply(calc_entropy)


def gen_feat(df, num_cols, cate_cols):
    fm = fill_miss_date(df, 'serial_number')
    df = fm.merge(df, on=['serial_number', 'dt'], how='left')
    df.sort_values(['serial_number', 'dt'], inplace=True)
    rolling_periods = [3, 5, 7]
    diff_periods = [2, 4, 6]
    # df[num_cols] = df.groupby('serial_number')[num_cols].fillna(method='ffill')
    for j in zip(rolling_periods, diff_periods):
        print(f'start gen feats, window size:diff-{j[1]}, rolling-{j[0]}')

        tmp_diff = df.groupby('serial_number')[num_cols].diff(j[1])
        tmp_diff.columns = [f'diff_{t}_{j[1]}' for t in num_cols]

        tmp_mean = df.groupby('serial_number')[num_cols].apply(get_rolling_mean, j[0])
        tmp_mean.columns = [f'mean_{t}_{j[1]}' for t in num_cols]

        tmp_std = df.groupby('serial_number')[num_cols].apply(get_rolling_std, j[0])
        tmp_std.columns = [f'std_{t}_{j[1]}' for t in num_cols]

        df = pd.concat([df, tmp_diff, tmp_mean, tmp_std], axis=1)

    tmp_nunique = df.groupby('serial_number')[cate_cols].apply(get_rolling_std, 7)
    tmp_nunique.columns = [f'nunique_{t}_7' for t in cate_cols]
    df = pd.concat([df, tmp_nunique], axis=1)

    for i in rolling_periods:
        print(f'start gen shift feats, window size: {i}')

        tmp_shift = df.groupby('serial_number')[cate_cols].shift(i)
        tmp_shift.columns = [f'shift_{t}_{i}' for t in cate_cols]

        df = pd.concat([df, tmp_shift], axis=1)

    df = df[df['model'].notnull()]
    return df


def gen_pos_label(tag_df, log_df):
    df = log_df.merge(tag_df[['serial_number', 'fault_time']], on='serial_number', how='left')
    df['label'] = (df['fault_time'] - df['dt']).dt.days
    df = df[df['label'] <= 30]
    df.loc[df['label'] < 0, 'label'] = 0
    return df


if __name__ == "__main__":
    time = datetime.datetime.now().strftime("%Y%m%d%H")
    data_path = 'tcdata'
    train_pos_path = f'{data_path}/{time}_train_positive_df.csv'
    train_neg_path = f'{data_path}/{time}_train_negative_df.csv'

    test_a = pd.read_csv(f'{data_path}/disk_sample_smart_log_test_a.csv')
    use_cols = ['model'] + [c for c in test_a.columns if not (test_a[c].isnull().sum()/test_a.shape[0] > 0.99
                                                              or test_a[c].nunique() < 2)]

    num_feats = [i for i in use_cols if test_a[i].value_counts(normalize=True).values[0] < 0.2 and i not in
                 ['serial_number', 'dt', 'model']]

    cate_feats = [i for i in use_cols if test_a[i].value_counts(normalize=True).values[0] >= 0.2 and i not in
                  ['serial_number', 'dt', 'model']]
    del test_a
    print(f'select {len(use_cols)} features ...')

    tag = pd.read_csv(f'{data_path}/disk_sample_fault_tag.csv')
    tag_201808 = pd.read_csv(f'{data_path}/disk_sample_fault_tag_201808.csv')
    del tag_201808['key']
    tag = tag.append(tag_201808)
    tag = tag.sort_values(['serial_number', 'fault_time', 'tag'])
    tag['serial_number'] = tag['model'].astype(str) + '_' + tag['serial_number']
    tag = tag.drop_duplicates(subset=['serial_number', 'fault_time'], keep='first')
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])

    if not os.path.exists(train_pos_path):
        train_pos = get_pos_df_from_tag(tag['serial_number'], use_cols, '2017-07-01', '2018-08-01')
        train_pos.to_csv(train_pos_path, index=False)
    else:
        train_pos = pd.read_csv(train_pos_path)

    if not os.path.exists(train_neg_path):
        train_neg = get_sample_df_for_negative(tag['serial_number'].unique(), 0.1,
                                               use_cols, '2018-04-01', '2018-08-01')
        train_neg.to_csv(train_neg_path, index=False)
    else:
        train_neg = pd.read_csv(train_neg_path)

    train_pos['dt'] = pd.to_datetime(train_pos['dt'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:]))

    train_neg['dt'] = pd.to_datetime(train_neg['dt'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:]))

    train_pos = gen_pos_label(tag, train_pos)
    print('train pos shape:', train_pos.shape)
    train_neg['label'] = 31
    train_neg['fault_time'] = np.nan
    print('train neg shape:', train_neg.shape)
    train = train_pos.append(train_neg)

    normalized_cols = [i for i in num_feats + cate_feats if 'normalized' in i]
    raw_cols = [i for i in num_feats + cate_feats if 'raw' in i]

    div_cols = []
    drop_cols = []
    for i in raw_cols:
        for j in normalized_cols:
            if i.split('raw') == j.split('_normalized'):
                tmp = i+'_div_'+j
                div_cols.append(tmp)
                drop_cols.extend([i, j])
                train[tmp] = train[i]/(train[j]+0.1)
                break
    train.drop(drop_cols, axis=1, inplace=True)
    num_feats = [i for i in num_feats if i not in drop_cols]
    cate_feats = [i for i in cate_feats if i not in drop_cols]
    num_feats.extend(div_cols)

    train = gen_feat(train, num_feats, cate_feats)
    train.to_csv(f'{data_path}/for_round2_train.csv', index=False)
