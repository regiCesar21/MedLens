import logging
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb

import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.expanduser("~/dataset/processed/")
PREDICTED_RES_DIR = os.path.join(PROCESSED_DIR, "predicted_results")

EXP_FILES_NUM = 5000


def get_random_files(hour):
    file_dir = os.path.join(PROCESSED_DIR, f"time_series_{hour}")
    files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
    files = random.sample(files, EXP_FILES_NUM)
    return files


def balanced_label_files(hours, exp_files_num):
    file_dir = os.path.join(PROCESSED_DIR, f"time_series_{hours[0]}")
    files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]

    with ThreadPoolExecutor(max_workers=12) as executor:
        file_paths = [os.path.join(file_dir, file) for file in files]
        label_futures = [executor.submit(get_label, file_path) for file_path in file_paths]

        label_results = []
        for future in tqdm(as_completed(label_futures), desc="Reading labels progress", total=len(label_futures),
                           unit='file'):
            label_results.append(future.result())

    file_label_pairs = list(zip(files, label_results))

    files_dict = {0: [], 1: []}

    for file, label in file_label_pairs:
        files_dict[label].append(file)
    logger.info(f"Number of files for label 0: {len(files_dict[0])}")
    logger.info(f"Number of files for label 1: {len(files_dict[1])}")

    num_files_per_label = min(exp_files_num // 2, len(files_dict[0]), len(files_dict[1]))
    if num_files_per_label * 2 < EXP_FILES_NUM:
        logger.info(f"Only {num_files_per_label * 2} available when requiring equal labels")

    selected_files_dict = {}
    for label, files in files_dict.items():
        selected_files = random.sample(files, num_files_per_label)
        selected_files_dict[label] = [file.rsplit('_', 1)[0] for file in selected_files]

    return selected_files_dict


def get_label(file_path):
    return int(pd.read_csv(file_path, usecols=["HOSPITAL_EXPIRE_FLAG"], nrows=1).values[0])


def load_single_file(file_path):
    df = pd.read_csv(file_path, index_col=None)
    df.drop("TIME\\ITEMID", axis=1, inplace=True)
    df.drop("SUBJECT_ID", axis=1, inplace=True)

    label = df["HOSPITAL_EXPIRE_FLAG"].values[0]  # Read HOSPITAL_EXPIRE_FLAG

    # filter numerical features
    numerical_item_ids = [str(x) for x in constants.TOP_LABEVENTS_ITEMID + constants.TOP_CHARTEVENTS_ITEMID_NUMERICAL]
    df = df[numerical_item_ids]
    time_series = df.loc[:, df.columns != "HOSPITAL_EXPIRE_FLAG"]  # Read ITEMID values
    time_series_compressed = pd.DataFrame([[time_series[col].values.tolist() for col in time_series.columns]])
    return time_series_compressed, label


def load_data(hour):
    file_dir = os.path.join(PROCESSED_DIR, f"time_series_{hour}")
    files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
    files = random.sample(files, EXP_FILES_NUM)

    with ThreadPoolExecutor(max_workers=12) as executor:
        file_paths = [os.path.join(file_dir, file) for file in files]
        futures = [executor.submit(load_single_file, file_path) for file_path in file_paths]

        results = [future.result() for future in
                   tqdm(as_completed(futures), desc="Loading data progress", total=len(files), unit='file')]

    time_series_list, labels = zip(*results)
    time_series_df = pd.concat(time_series_list, ignore_index=True)
    exclude_columns = ["TIME\\ITEMID", "SUBJECT_ID", "HOSPITAL_EXPIRE_FLAG"]

    numerical_item_ids = [str(x) for x in constants.TOP_LABEVENTS_ITEMID + constants.TOP_CHARTEVENTS_ITEMID_NUMERICAL]
    df = pd.read_csv(os.path.join(file_dir, files[0]), index_col=None, nrows=0).drop(
        columns=exclude_columns)
    num_df = df[numerical_item_ids]

    time_series_df.columns = num_df.columns
    return time_series_df, np.array(labels)


def flatten_dataframe(df):
    flattened_data = []
    for index, row in df.iterrows():
        flat_row = [item for cell in row for item in cell]
        flattened_data.append(flat_row)
    return np.array(flattened_data)


def train_and_evaluate_classifier(name, clf, X_train, X_test, y_train, y_test, hour, RF_interpolation):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    y_score = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_score)
    auprc = average_precision_score(y_test, y_score)

    logger.info(f"\n{name} - Accuracy: {accuracy}, F1 Score: {f1}, AUC-ROC: {auroc}, AUC-PR: {auprc}")
    return hour, RF_interpolation, name, accuracy, f1, auroc, auprc


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, hour, RF_interpolation):
    classifiers = [
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("MLP Classifier", MLPClassifier(random_state=42)),
        ("Logistic Regression", LogisticRegression(random_state=42)),
        ("XGBoost", xgb.XGBClassifier(random_state=42)),
        ("LightGBM", lgb.LGBMClassifier(random_state=42)),
    ]

    results = []
    with ThreadPoolExecutor(max_workers=len(classifiers)) as executor:
        futures = [executor.submit(train_and_evaluate_classifier, name, clf, X_train, X_test, y_train, y_test, hour,
                                   RF_interpolation) for name, clf in classifiers]

        for future in tqdm(as_completed(futures), total=len(classifiers),
                           desc="Classifier training progress"):
            results.append(future.result())

    return results


def interpolation_base(df):
    for column in df.columns:
        for i in range(len(df)):
            s = pd.Series(df[column].values[i]).ffill().bfill().fillna(0.0)
            df[column].values[i] = s.values
    return df


def get_instance_features(ts, p, w):
    funcs = ['max', 'min', 'mean', 'diff', 'pos']
    st = max(0, p - w)
    ed = min(len(ts) - 1, p + w)
    sub_ts = ts[st: ed + 1]
    sub_ts = [x for x in sub_ts if not np.isnan(x)]
    if len(sub_ts) == 0: return None
    xs = []
    for func in funcs:
        if func == 'max': xs.append(max(sub_ts))
        if func == 'min': xs.append(min(sub_ts))
        if func == 'mean': xs.append(np.mean(sub_ts))
        if func == 'diff': xs.append(max(sub_ts) - min(sub_ts))
        if func == 'pos': xs.append(p % 24)
    return xs


def batch_extract_features(list_of_ts, window):  # random forest regression: mean, max, min, std, diff
    X = []
    y = []
    for time_series in list_of_ts:
        for i in range(len(time_series)):
            if np.isnan(time_series[i]): continue
            xs = get_instance_features(time_series, i, window)
            if xs is not None: X.append(xs)
            y.append(time_series[i])
    return X, y


def transform_a_ts(rf, ts, window_size):
    new_ts = []
    for i in range(len(ts)):
        if not np.isnan(ts[i]):
            new_ts.append(ts[i])
        else:
            xs = get_instance_features(ts, i, window_size)
            if xs is None:
                new_ts.append(ts[i])
            else:
                new_ts.append(rf.predict([xs])[0])
    return new_ts


def interpolation_regression(df_train, df_test, column, window_size=12):
    train_X, train_y = batch_extract_features(df_train[column].values, window_size)
    rf = RandomForestRegressor()
    rf.fit(train_X, train_y)
    new_ts_list = []
    for ts in df_test[column].values:
        new_ts = transform_a_ts(rf, ts, window_size)
        new_ts_list.append(new_ts)
    df_test[column] = new_ts_list
    return df_test


def main(hour):
    random.seed(123)
    logger.info(f"Balanced label files ...")
    files_dict = balanced_label_files([hour], exp_files_num=EXP_FILES_NUM)
    logger.info(f"Loading data ...")
    X, y = load_data(hour)
    X = interpolation_base(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Running interpolation for {hour} ...")
    for column in tqdm(X_train.columns, desc="Interpolation progress"):
        X_train = interpolation_regression(X_train, X_train, column)
        X_test = interpolation_regression(X_test, X_test, column)

    logger.info(f"Training and evaluating classifiers ...")
    results = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, hour, True)

    results_df = pd.DataFrame(results, columns=['Hour', 'RF Interpolation', 'Classifier', 'Accuracy', 'F1 Score',
                                                'AUC-ROC', 'AUC-PR'])
    results_df.to_csv(os.path.join(PREDICTED_RES_DIR, f"predicted_results_{hour}.csv"), index=False)
    return results_df


if __name__ == "__main__":
    hours = [720, 360, 180]
    for hour in hours:
        main(hour)
