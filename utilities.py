"""
Utilities for ML model fitting.

Jamie Taylor
Fariba Yousefi
2022-03-18
"""

import numpy as np
import pandas as pd
from dask.dataframe import from_pandas
np.random.seed(42)


def expand_1hot_cols_alt(df, cat_cols, drop_og_cols_from_df=False):
    # N.B. Using pandas `get_dummies()` uses a lot of RAM due to the need for joins
    # N.B. Use DASK for faster (parallel) compute
    df_ = from_pandas(df[cat_cols].copy(), npartitions=8)
    new_cat_cols = []
    for col in cat_cols:
        for val in df_[col].unique():
            col_1hot = f"{col}/{val}"
            df_[col_1hot] = df_[col].apply(lambda r: 1 if r == val else 0).astype(np.int8)
            new_cat_cols.append(col_1hot)
    df_ = df_.compute()
    df = pd.concat([df, df_], axis=1, ignore_index=True)
    if drop_og_cols_from_df:
        df.drop(columns=cat_cols, inplace=True)
    return df, new_cat_cols

def expand_1hot_cols(df, cat_cols, drop_og_cols_from_df=False):
    cat_cols_ = cat_cols.copy()
    for i, col in enumerate(cat_cols):
        print(f"Expanding '{col}' ({i+1} of {len(cat_cols)})...")
        tmp = pd.get_dummies(df[col], prefix=f"{col}", prefix_sep="/")
        df = df.join(tmp)
        if drop_og_cols_from_df:
            df.drop(columns=col, inplace=True)
        cat_cols_.remove(col)
        cat_cols_ += list(tmp.columns)
        print(f"    -> added {len(list(tmp.columns))} new cols")
    return df, cat_cols_

def prepare_xy(df, num_cols, cat_cols, cyc_cols, target_col):
    x = df[num_cols + cat_cols + cyc_cols].copy()
    y = df[target_col].copy()
    return x, y

def encode_cyclical_cols(df, cyc_cols, max_vals, drop_og_cols_from_df=False):
    cyc_cols_ = cyc_cols.copy()
    for i, col in enumerate(cyc_cols):
        print(f"Applying cyclical encoding to '{col}' ({i+1} of {len(cyc_cols)})...")
        df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_vals[i])
        df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_vals[i])
        cyc_cols_ += [col + "_sin", col + "_cos"]
        cyc_cols_.remove(col)
    if drop_og_cols_from_df:
        df.drop(columns=cyc_cols, inplace=True)
    return df, cyc_cols_

def train_test_split_by_col(df, train_ratio, valid_ratio, test_ratio, by_col):
    assert np.allclose(train_ratio + valid_ratio + test_ratio, 1)
    unique_ids = df[by_col].unique()
    train_count = int(len(unique_ids) * train_ratio)
    valid_count = int(len(unique_ids) * valid_ratio)
    train_ids = np.random.choice(unique_ids, size=train_count, replace=False)
    valid_ids = np.random.choice([i for i in unique_ids if i not in train_ids], size=valid_count,
                                 replace=False)
    test_ids = [i for i in unique_ids if i not in np.concatenate((train_ids, valid_ids))]
    df["set"] = ""
    df.loc[df[by_col].isin(train_ids), ("set")] = "train"
    df.loc[df[by_col].isin(valid_ids), ("set")] = "valid"
    df.loc[df[by_col].isin(test_ids), ("set")] = "test"
    df.loc[:, "set"] = df.loc[:, "set"].astype("category")
    return df

def normalise_cols(df, cols):
    x_train_mean = df.loc[df.set=="train", cols].mean()
    x_train_std = df.loc[df.set=="train", cols].std()
    df.loc[:, cols] = (df.loc[:, cols] - x_train_mean) / x_train_std
    return df, x_train_mean, x_train_std

def unnormalise_cols(df, cols, mean, std):
    df[cols] = (df[cols] * std) / mean
    return df

def remove_anomalous_gsps(df, threshold_p=0.8):
    df, n = remove_gsps_with_negative_flows(df)
    df, c = remove_gsps_not_correlated_with_pes(df, threshold_p)
    return df, n, c

def remove_gsps_with_negative_flows(df):
    gsp_meter_vol_plus_pv = df["gsp_meter_volume"] + df["pv_generation_mw"] * 1.1
    neg_regions = df.loc[gsp_meter_vol_plus_pv < 0, "region_id_20210423"].unique()
    df = df[~df["region_id_20210423"].isin(neg_regions)]
    return df, len(neg_regions)

def remove_gsps_not_correlated_with_pes(df, threshold_p):
    corr = df.groupby("region_id_20210423")[["pes_meter_volume", "gsp_meter_volume"]].corr()
    bad_ids = corr.loc[corr.gsp_meter_volume < threshold_p].index.get_level_values(0)
    df = df[~df.region_id_20210423.isin(bad_ids)]
    return df, len(bad_ids)