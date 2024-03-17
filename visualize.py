import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_age_distribution(df):
    sns.set_context("paper", font_scale=1, rc={"font.size": 3, "axes.titlesize": 15, "axes.labelsize": 10})
    ax = sns.catplot(kind='count', data=df, x='age', hue='target', order=df['age'].sort_values().unique())
    ax.ax.set_xticks(np.arange(0, 80, 5))
    plt.title('Variation of Age for each target class')
    plt.show()


def plot_age_sex_distribution(df):
    sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
    plt.title('Distribution of age vs sex with the target class')
    plt.show()