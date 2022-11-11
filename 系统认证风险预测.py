import warnings

warnings.simplefilter('ignore')
import gc
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 读取数据
train = pd.read_csv('data/train_dataset.csv', sep='\t')
test = pd.read_csv('data/test_dataset.csv', sep='\t')
data = pd.concat([train, test], axis=0)

# location列转成多列
data['location_first_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
data['location_sec_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
data['location_third_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

# 删除值相同的列
data.drop(['client_type', 'browser_source'], axis=1, inplace=True)
data['auth_type'].fillna('__NaN__', inplace=True)

# 日期数据处理
data['op_date'] = pd.to_datetime(data['op_date'])
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
data['ts_diff1'] = data['op_ts'] - data['last_ts']

data['op_date_month'] = data['op_date'].dt.month  # 月份放到年份的前面居然上分
data['op_date_year'] = data['op_date'].dt.year
data['op_date_day'] = data['op_date'].dt.day
data['op_date_dayofweek'] = data['op_date'].dt.dayofweek
data['op_date_ymd'] = data['op_date'].dt.year * 100 + data['op_date'].dt.month
data['op_date_hour'] = data['op_date'].dt.hour

period_dict = {
    23: 0, 0: 0, 1: 0,
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,
    13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
    19: 4, 20: 4, 21: 4, 22: 4,
}
data['hour_cut'] = data['op_date_hour'].map(period_dict)
# 一年中的哪个季度
season_dict = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
    10: 4, 11: 4, 12: 4,
}
data['month_cut'] = data['op_date_month'].map(season_dict)
data['dayofyear'] = data['op_date'].apply(lambda x: x.dayofyear)  # 一年中的第几天
data['weekofyear'] = data['op_date'].apply(lambda x: x.week)  # 一年中的第几周
data['是否周末'] = data['op_date'].apply(lambda x: True if x.dayofweek in [4, 5, 6] else False)  # 是否周末
data.loc[((data['op_date_hour'] >= 7) & (data['op_date_hour'] < 22)), 'isworktime'] = 1

# 特征编码
data['ip_risk_level'] = data['ip_risk_level'].map({'1级': 1, '2级': 2, '3级': 3})

for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
    data[f'user_{f}_nunique'] = data.groupby(['user_name'])[f].transform('nunique')

for method in ['mean', 'max', 'min', 'std']:
    data[f'ts_diff1_{method}'] = data.groupby('user_name')['ts_diff1'].transform(method)

for i in ['os_type']:
    data[i + '_n'] = data.groupby(['user_name', 'op_date_ymd', 'op_date_hour'])[i].transform('nunique')

lis = ['user_name', 'action',
       'auth_type',
       'ip',
       'ip_location_type_keyword', 'device_model',
       'os_type', 'os_version', 'browser_type', 'browser_version',
       'bus_system_code', 'op_target', 'location_first_lvl', 'location_sec_lvl',
       'location_third_lvl']
# one_hot
data_re = data[lis]
df_processed = pd.get_dummies(data_re, prefix_sep="_", columns=data_re.columns)
lis_sx = [i for i in data.columns if i not in lis]
data = pd.concat([data[lis_sx], df_processed], axis=1)
train = data[data['risk_label'].notna()]
test = data[data['risk_label'].isna()]


ycol = 'risk_label'

feature_names = list(
    filter(lambda x: x not in [ycol, 'session_id', 'op_date', 'location', 'last_ts'], train.columns))

model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=29,  # 29
                           max_depth=7,  # 7
                           learning_rate=0.07,  # 0.07
                           n_estimators=2000,  # 1950
                           subsample=0.7,  # 0.7
                           feature_fraction=0.95,  # 0.95
                           reg_alpha=0.,
                           reg_lambda=0.,
                           random_state=1973,  # 1973
                           is_unbalance=True,
                           metric='auc')

oof = []
prediction = test[['session_id']]
prediction[ycol] = 0
df_importance_list = []

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1950)  # 1950
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]

    print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=300,  # 250
                          eval_metric='auc',
                          early_stopping_rounds=500)  # 400

    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)
    df_oof = train.iloc[val_idx][['session_id', ycol]].copy()
    df_oof['pred'] = pred_val[:, 1]
    oof.append(df_oof)

    pred_test = lgb_model.predict_proba(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    prediction[ycol] += pred_test[:, 1] / kfold.n_splits

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
    gc.collect()

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()
df_importance

df_oof = pd.concat(oof)
print('roc_auc_score', roc_auc_score(df_oof[ycol], df_oof['pred']))


prediction['id'] = range(len(prediction))
prediction['id'] = prediction['id'] + 1
prediction = prediction[['id', 'risk_label']].copy()
prediction.columns = ['id', 'ret']
prediction.to_csv('result/res.csv', index=False)
