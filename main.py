#!/usr/bin/env python
# -*- coding:utf-8 -*-

from config import config
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import zipfile
import glob

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


def get_rolling_mean(grp, freq):
    return grp.rolling(freq).mean()


def get_rolling_std(grp, freq):
    return grp.rolling(freq).std()


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


def lgb_predict(test_, feats, threshold):
    sub_preds = np.zeros((test_.shape[0], 5))
    for i in range(1, 6):
        clf = lgb.Booster(model_file=f'pakdd_lgb_{i}.txt')
        sub_preds[:, i - 1] = clf.predict(test_[feats], num_iteration=clf.best_iteration)
    test_['pred'] = np.mean(sub_preds, axis=1)
    test_['rank'] = test_.groupby('dt')['pred'].rank(method='min')
    test_ = test_[test_['dt'] >= '2018-09-01']
    test_['day'] = test_['dt'].dt.day
    test_['day'] = test_['day'].map(day_map)

    for_sub = test_[(test_['rank'] <= test_['day']) & (test_['pred'] <= threshold)]

    for_sub.sort_values(['serial_number', 'dt'], inplace=True)
    for_sub = for_sub.drop_duplicates(subset=['serial_number'], keep='first')
    print('Total predict cnt:', for_sub.shape[0])
    for_sub['manufacturer'] = 'A'
    for_sub['model'] = for_sub['serial_number'].apply(lambda x: x[0])
    for_sub['serial_number'] = for_sub['serial_number'].apply(lambda x: x[2:])
    for_sub[['manufacturer', 'model', 'serial_number', 'dt']].to_csv('result.csv', index=False, header=None)
    z = zipfile.ZipFile(SUB_FILE, 'w')
    z.write('result.csv', compress_type=zipfile.ZIP_DEFLATED)
    z.close()


if __name__ == "__main__":
    SUB_FILE = config.sub_file
    TEST_FILE = config.test_dir
    read_cols = ['model', 'serial_number', 'smart_1_normalized', 'smart_1raw', 'smart_3_normalized',
                 'smart_4_normalized', 'smart_4raw', 'smart_5_normalized', 'smart_5raw', 'smart_7_normalized',
                 'smart_7raw', 'smart_9_normalized', 'smart_9raw', 'smart_12_normalized', 'smart_12raw',
                 'smart_184_normalized', 'smart_184raw', 'smart_187_normalized', 'smart_187raw',
                 'smart_188_normalized', 'smart_188raw', 'smart_189_normalized', 'smart_189raw', 'smart_190_normalized',
                 'smart_190raw', 'smart_192_normalized', 'smart_192raw', 'smart_193_normalized', 'smart_193raw',
                 'smart_194_normalized', 'smart_194raw', 'smart_195_normalized', 'smart_195raw', 'smart_197_normalized',
                 'smart_197raw', 'smart_198_normalized', 'smart_198raw', 'smart_199raw', 'dt']

    day_map = {k: (k//11+3)*8 for k in range(1, 31)}

    test_list = glob.glob('%s/*.csv' % TEST_FILE)
    print('loading test file ...')
    test_b = pd.concat([pd.read_csv(t, usecols=read_cols) for t in test_list], axis=0)

    test_b['dt'] = pd.to_datetime(test_b['dt'].astype(str).apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:]))
    test_b['serial_number'] = test_b['model'].astype(str) + '_' + test_b['serial_number']
    num_feats = ['smart_1raw', 'smart_4raw', 'smart_7_normalized', 'smart_7raw',
                 'smart_12raw', 'smart_190_normalized', 'smart_190raw', 'smart_192raw', 'smart_193raw',
                 'smart_194_normalized', 'smart_194raw', 'smart_195_normalized', 'smart_195raw']
    cate_feats = ['smart_1_normalized', 'smart_3_normalized', 'smart_4_normalized', 'smart_5_normalized', 'smart_5raw',
                  'smart_12_normalized', 'smart_184_normalized', 'smart_184raw', 'smart_187_normalized', 'smart_187raw',
                  'smart_188_normalized', 'smart_188raw', 'smart_189_normalized', 'smart_189raw', 'smart_192_normalized',
                  'smart_193_normalized', 'smart_197_normalized', 'smart_197raw', 'smart_198_normalized', 'smart_198raw',
                  'smart_199raw']

    normalized_cols = [i for i in num_feats + cate_feats if 'normalized' in i]
    raw_cols = [i for i in num_feats + cate_feats if 'raw' in i]

    div_cols = []
    drop_cols = ['smart_9_normalized', 'smart_9raw']
    for i in raw_cols:
        for j in normalized_cols:
            if i.split('raw') == j.split('_normalized'):
                tmp = i+'_div_'+j
                div_cols.append(tmp)
                drop_cols.extend([i, j])
                test_b[tmp] = test_b[i]/(test_b[j]+0.1)
                break

    test_b.drop(drop_cols, axis=1, inplace=True)
    num_feats = [i for i in num_feats if i not in drop_cols]
    cate_feats = [i for i in cate_feats if i not in drop_cols]
    num_feats.extend(div_cols)

    for_test = gen_feat(test_b, num_feats, cate_feats)
    print('features finished.')
    use_cols = [c for c in for_test.columns if c not in ['serial_number', 'dt', 'label', 'fault_time']]
    lgb_predict(for_test, use_cols, 15)
    print('result generated.')
