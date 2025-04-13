import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, RobustScaler, MinMaxScaler

def select_relevant_features(full_df):
    df = full_df[['title', 'publication_year', 'cited_by_count', 'topics.display_name', 'topics.score']].copy()
    df['topics.score'] = df['topics.score'].astype('object').str.split('|').apply(
        lambda x: list(map(float, x)) if isinstance(x, list) else []
    )

    df['topics.display_name'] = df['topics.display_name'].astype('object').str.split('|')
    df.dropna(subset=['title'], inplace=True)
    return df

def normalize_numerical_features(df):
    year_min = df['publication_year'].min()
    scaler = MinMaxScaler()
    df['publication_year_norm'] = scaler.fit_transform(pd.DataFrame(np.log1p(df['publication_year'] - year_min + 1)))

    year_min = df['cited_by_count'].min()
    scaler = MinMaxScaler()
    df['cited_by_count_norm'] = scaler.fit_transform(pd.DataFrame(np.log1p(df['cited_by_count'] - year_min + 1)))

    return df

def generate_topic_indices(topics_list, mapping):
    """
    Converts a list of topic strings to a list of indices using the provided mapping.
    If a topic is not found, it defaults to the padding index.
    """
    return [mapping.get(topic, mapping["<PAD>"]) for topic in topics_list]

def clean_dataset(full_df):
    df = select_relevant_features(full_df)
    df = normalize_numerical_features(df)
    return df

def pad_topic_list(topics, max_len=3, pad_token="<PAD>"):
    pad_length = max_len - len(topics)
    if pad_length > 0:
        topics.extend([pad_token] * pad_length)
    return topics

def pad_score_list(scores, max_len=3, pad_value=0.0):
    pad_length = max_len - len(scores)
    if pad_length > 0:
        scores.extend([pad_token] * pad_length)
    return scores

def export_df(df, file_name):
    df.to_csv(file_name, header=True, index=False)


