import string
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

from collections import defaultdict
from wordcloud import STOPWORDS

"""
Data Exploration for the Jigsaw Competition

Table of contents:
1. N-Grams
2. BoxPlots
3. Distribution
"""

DATA_PATH_COMMENTS = "../data/jigsaw-toxic-severity-rating/comments_to_score.csv"
DATA_PATH_VALIDATION = "../data/jigsaw-toxic-severity-rating/validation_data.csv"
comments_df = pd.read_csv(DATA_PATH_COMMENTS)
validation_df = pd.read_csv(DATA_PATH_VALIDATION)

"""
1. N-Gram (sometimes also called Q-gram) is a contiguous 
sequence of n items from a given sample of text or speech
"""
def _generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    n_grams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in n_grams]


def binarize_ngrams(target_df, target_column_name, n_gram=1, N=20):
    target_ngrams = defaultdict(int)
    for comment in target_df[target_column_name]:
        for word in _generate_ngrams(comment, n_gram):
            target_ngrams[word] += 1

    df_target_ngrams = pd.DataFrame(sorted(target_ngrams.items(), key=lambda x: x[1])[::-1])
    ngrams_bounded = df_target_ngrams[:N]
    return ngrams_bounded


def plot_ngram(target_ngram):
    # to note: call this method with the result from binarize_ngrams
    fig, axes = plt.subplots(ncols=1, figsize=(18, len(target_ngram) // 2), dpi=100)
    plt.tight_layout()
    sns.barplot(y=target_ngram[0], x=target_ngram[1], ax=axes, color='green')

    axes.spines['right'].set_visible(False)
    axes.set_xlabel('')
    axes.set_ylabel('')
    axes.tick_params(axis='x', labelsize=13)
    axes.tick_params(axis='y', labelsize=13)

    axes.set_title(f'Top {len(target_ngram)} most common ngrams in the given comments', fontsize=15)
    plt.show()


def plot_two_ngrams(target_ngram1, target_ngram2):
    fig, axes = plt.subplots(ncols=2, figsize=(30, len(target_ngram1) // 2), dpi=100)
    plt.tight_layout()

    sns.barplot(y=target_ngram1[0], x=target_ngram1[1], ax=axes[0], color='green')
    sns.barplot(y=target_ngram2[0], x=target_ngram2[1], ax=axes[1], color='red')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=13)

    axes[0].set_title(f'Top {len(target_ngram2)} most common trigrams in first ngram', fontsize=35)
    axes[1].set_title(f'Top {len(target_ngram2)} most common trigrams in second ngram', fontsize=35)

    plt.show()


"""
2. BoxPlots
"""
def plot_number_of_words(target_df, target_column_name):
    count_target_words = [len(sentence.split(' ')) for sentence in target_df[target_column_name].values]
    _draw_boxplot(count_target_words, target_column_name)


def plot_number_of_char(target_df, target_column_name):
    count_target_chars = [len(sentence) for sentence in target_df[target_column_name].values]
    _draw_boxplot(count_target_chars, target_column_name)


def plot_number_of_punct(target_df, target_column_name):
    count_target_punct = [len([char for char in sentence if char in string.punctuation]) for sentence in
                          target_df[target_column_name].values]
    _draw_boxplot(count_target_punct, target_column_name)


def _draw_boxplot(count, name):
    # Note: the resulting plot will appear in your browser
    fig = go.Figure()
    fig.add_trace(go.Box(y=count, name=name, ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="LightSteelBlue",
    )
    fig.show()


"""
3. Distribution
"""
def show_distribution_words(target_df, target_column_name, sample=5000):
    count_target_words = [len(sentence.split(' ')) for sentence in target_df[target_column_name].values]
    sample_words = random.sample(count_target_words, sample)
    hist_data = [sample_words]
    X = [target_column_name]
    fig = ff.create_distplot(hist_data, X, show_hist=False)
    fig.show()


def show_two_word_distributions(target_df1, target_column_name1, target_df2, target_column_name2, sample=5000):
    count_target_words1 = [len(sentence.split(' ')) for sentence in target_df1[target_column_name1].values]
    count_target_words2 = [len(sentence.split(' ')) for sentence in target_df2[target_column_name2].values]
    sample_words1 = random.sample(count_target_words1, sample)
    sample_words2 = random.sample(count_target_words2, sample)
    hist_data = [sample_words1, sample_words2]
    X = [target_column_name1, target_column_name2]
    fig = ff.create_distplot(hist_data, X, show_hist=False)
    fig.show()


if __name__ == '__main__':
    print("Running data analysis")
    # insert here what plotting needs to be done
    print("Finished running data analaysis")
