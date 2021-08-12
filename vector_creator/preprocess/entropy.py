import vector_creator.stats_models.estimators as est
import numpy as np

def entropy_of_duration_by_cat(df, dur_col, cat_col, cat):
    df = df.loc[df[cat_col] == cat]
    y =  df[dur_col].to_numpy()
    y0 = 10 * np.round(y.astype(np.float64)/10, 0)
    return [est.entropy(y0)]


# for call_logs, freq column is phone numbers
def entropy_of_freq_by_cat(df, freq_col, cat_col, cat):
    df = df.loc[df[cat_col] == cat]
    y = df.groupby(freq_col)[freq_col].agg('count').to_numpy()
    y0 = 10 * np.round(y.astype(np.float64)/10, 0)
    return [est.entropy(y0)]


def entropy_of_cat(df, cat_col, categories):
    def count_occurrence_by_cat(df0, c_col, cats):
        occurr = []
        for cat in cats:
            occurr.append(df0[df0[c_col] == cat].shape[0])
        return np.array(occurr)
    df[cat_col] = df[cat_col].apply(lambda x : x.split('/')[1] if len(x.split('/')) == 2 else x)
    y = count_occurrence_by_cat(df, cat_col, categories)
    return [est.entropy(y)]
