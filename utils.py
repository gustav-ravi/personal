from google.cloud import translate_v2
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from numbers import Number
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import emoji
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import spacy
import string
import typing
from typing import List, Optional, Tuple, Union, Dict

punctuation="".join(list(filter(lambda x: x!="_", string.punctuation)))

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the columns of a DataFrame by converting them to lowercase,
    replacing spaces with underscores, and handling duplicate column names.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns to be renamed.

    Returns:
    pd.DataFrame: The DataFrame with columns renamed according to the specified rules.
    """
    columns=df.columns
    new_columns=[i.lower().replace(" ","_") for i in columns]
    new_columns=[i.translate(str.maketrans({i:"_" for i in punctuation})) for i in new_columns]
    new_columns=[re.sub("_+", "_", i) for i in new_columns]
    new_columns=[i[:-1] if i[-1]=="_" else i for i in new_columns]
    if len(set(new_columns))<len(new_columns):
        column_change_dict=dict(zip(columns, new_columns))
        df.rename(columns=column_change_dict, inplace=True)
    else:
        index=2
        column_set=["_".join(column.split("_")[:min(len(column.split("_")),index)]) for column in new_columns]
        while len(set(column_set))<len(column_set):
            index+=1
            column_set=["_".join(column.split("_")[:min(len(column.split("_")),index)]) for column in new_columns]
        column_change_dict=dict(zip(columns, column_set))
        df.rename(columns=column_change_dict, inplace=True)
    return df

def add_value_labels(ax: plt.Axes, spacing: int = 5, labels: Optional[List[str]] = None, horizontal: bool = False) -> None:
    """
    Add labels with values on a matplotlib bar plot.

    Parameters:
    - ax (plt.Axes): The matplotlib Axes object representing the bar plot.
    - spacing (int, optional): The spacing between the bar and the label. Default is 5.
    - labels (List[str], optional): List of labels to be added to the bars. If None, numeric values are used.
    - horizontal (bool): If True, labels are added horizontally; otherwise, vertically.

    Returns:
    None
    """
    if labels is None:
        if horizontal:
            for i, value in enumerate(ax.patches):
                ax.annotate(
                value.get_width(),                      
                (value.get_width(),i),         
                xytext=(5,0),          
                textcoords="offset points", 
                ha='left',                
                va='center')
        else: 
            for rect in ax.patches:
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2
                space = spacing
                va = 'bottom'
                if y_value < 0:
                    space *= -1
                    va = 'top'
                if y_value-int(y_value)==0:
                    y_value=int(y_value)
                    label=str(y_value)
                else:
                    label = "{:.3f} m".format(y_value)
                ax.annotate(
                    label,                      
                    (x_value, y_value),         
                    xytext=(0, space),          
                    textcoords="offset points", 
                    ha='center',                
                    va=va)    
    else:
        if horizontal:
            for i, value in enumerate(ax.patches):
                ax.annotate(
                labels[i],                      
                (value.get_width(),i),         
                xytext=(5,0),          
                textcoords="offset points", 
                ha='left',                
                va='center') 
        else:       
            for idx, rect in enumerate(ax.patches):
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2
                space = spacing
                va = 'bottom'
                if y_value < 0:
                    space *= -1
                    va = 'top'
                label = str(labels[idx])
                ax.annotate(
                    label,                      
                    (x_value, y_value),         
                    xytext=(0, space),          
                    textcoords="offset points", 
                    ha='center',                
                    va=va)         
            

def set_appearance(ax: plt.Axes, labels: List[str], ylabel: Optional[str] = None, xlabel: Optional[str] = None,
                    title: Optional[str] = None, use_labels: bool = True, label_direction: str = "x",
                    axes: str = "both", no_ticks: str = "both") -> plt.Axes:
    """
    Customize the appearance of a matplotlib Axes object.

    Parameters:
    - ax (plt.Axes): The matplotlib Axes object to be customized.
    - labels (List[str]): List of labels for ticks.
    - ylabel (str, optional): Label for the y-axis.
    - xlabel (str, optional): Label for the x-axis.
    - title (str, optional): Title for the plot.
    - use_labels (bool): If True, use labels for ticks; otherwise, use numeric ticks.
    - label_direction (str): Direction of labels, "x" for x-axis, "y" for y-axis.
    - axes (str): Axis to apply the grid ("both", "x", "y").
    - no_ticks (str): Axes from which ticks will be removed ("both", "x", "y").

    Returns:
    plt.Axes: The customized matplotlib Axes object.
    """

    ax.set_title(title, weight="bold",size=15)
    ax.set_axisbelow(True)
    
    if use_labels:
        if label_direction=="x":
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
        if label_direction=="y":
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    ax.grid(linestyle="--", axis=axes)
    ax.tick_params(length=0, axis=no_ticks)
    ax.set_ylabel(ylabel,size=13,weight="bold")
    ax.set_xlabel(xlabel,size=13,weight="bold")
    return ax


def bar_plot(values: List[Union[int, float]], labels: List[str], color: str = "black", title: Optional[str] = None,
             title_suffix: Optional[str] = None, size: tuple = (10, 7), ylabel: Optional[str] = None,
             xlabel: Optional[str] = None, annotate: bool = False, rotate_labels: bool = False,
             highlight_best: bool = False, return_ax: bool = False, save: bool = False) -> Optional[plt.Axes]:
    """
    Create a bar plot with customization options.

    Parameters:
    - values (List[Union[int, float]]): List of values for the bars.
    - labels (List[str]): List of labels for the bars.
    - color (str, optional): Color of the bars. Default is "black".
    - title (str, optional): Title for the plot.
    - title_suffix (str, optional): Suffix to be added to the title.
    - size (tuple, optional): Size of the figure. Default is (10, 7).
    - ylabel (str, optional): Label for the y-axis.
    - xlabel (str, optional): Label for the x-axis.
    - annotate (bool): If True, add value labels to the bars.
    - rotate_labels (bool): If True, rotate x-axis labels.
    - highlight_best (bool): If True, highlight the bar with the highest value.
    - return_ax (bool): If True, return the matplotlib Axes object.
    - save (bool): If True, save the plot as an image file.

    Returns:
    Optional[plt.Axes]: The matplotlib Axes object if return_ax is True, otherwise None.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(size)
    ax.bar(range(len(labels)), values,color=color)
    if highlight_best:
        ax.bar(values.index(max(values)), max(values), color="springgreen", label="best")
    ax.set_xticks(range(len(labels)))
    if rotate_labels:
        ax.set_xticklabels(labels,rotation=90,size=12)
    else:
        ax.set_xticklabels(labels, size=12)
    if highlight_best:
        ax.legend()
    if annotate:
        add_value_labels(ax)
    ax=set_appearance(ax, labels, xlabel=xlabel, ylabel=ylabel, title=title, axes="y", no_ticks="y", label_direction="x", use_labels=True)
    if return_ax:
        return ax
    if save:
        if title is None:
            title = "no_title"
        title=retitle(title)
        try:
            plt.savefig(title, bbox_inches='tight')
        except:
            plt.savefig(title.replace("\n","_"), bbox_inches="tight")
    else:
        plt.show()

def barh_plot(values: List[Union[int, float]], labels: List[str], color: str = "black", title: Optional[str] = None,
              title_suffix: Optional[str] = None, size: tuple = (10, 7), ylabel: Optional[str] = None,
              xlabel: Optional[str] = None, annotate: bool = False, return_ax: bool = False,
              highlight_best: bool = False, save: bool = False) -> Optional[plt.Axes]:
    """
    Create a horizontal bar plot with customization options.

    Parameters:
    - values (List[Union[int, float]]): List of values for the bars.
    - labels (List[str]): List of labels for the bars.
    - color (str, optional): Color of the bars. Default is "black".
    - title (str, optional): Title for the plot.
    - title_suffix (str, optional): Suffix to be added to the title.
    - size (tuple, optional): Size of the figure. Default is (10, 7).
    - ylabel (str, optional): Label for the y-axis.
    - xlabel (str, optional): Label for the x-axis.
    - annotate (bool): If True, add value labels to the bars.
    - return_ax (bool): If True, return the matplotlib Axes object.
    - highlight_best (bool): If True, highlight the bar with the highest value.
    - save (bool): If True, save the plot as an image file.

    Returns:
    Optional[plt.Axes]: The matplotlib Axes object if return_ax is True, otherwise None.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(size)
    ax.barh(range(len(labels)), values,color=color)
    if highlight_best:
        ax.barh(values.index(max(values)), max(values), color="springgreen", label="best")
    if annotate:
        for i, value in enumerate(values):
            ax.annotate(
            value,                      
            (value,i),         
            xytext=(5,0),          
            textcoords="offset points", 
            ha='left',                
            va='center')  
    ax.set_title(title, weight="bold")
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    ax.grid(linestyle="--", axis="x")
    ax.tick_params(length=0, axis="x")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if return_ax:
        return ax
    if save:
        title=retitle(title)
        if title_suffix is not None:
            title+=title_suffix
        plt.savefig(title+".png",bbox_inches='tight')
    else:
        plt.show()


def line_plot(values: List[Union[int, float]], labels: List[Union[int, float]], color: str = "black", scatter: bool = False,
              title: Optional[str] = None, title_suffix: Optional[str] = None, size: tuple = (10, 7),
              ylabel: Optional[str] = None, xlabel: Optional[str] = None, label: Optional[str] = None,
              highlight_best: bool = False, perc: bool = False, save: bool = False, return_ax: bool = False) -> Optional[plt.Axes]:
    """
    Create a line plot with customization options.

    Parameters:
    - values (List[Union[int, float]]): List of values for the plot.
    - labels (List[Union[int, float]]): List of labels for the plot.
    - color (str, optional): Color of the line. Default is "black".
    - scatter (bool): If True, create a line plot with scatter points.
    - title (str, optional): Title for the plot.
    - title_suffix (str, optional): Suffix to be added to the title.
    - size (tuple, optional): Size of the figure. Default is (10, 7).
    - ylabel (str, optional): Label for the y-axis.
    - xlabel (str, optional): Label for the x-axis.
    - label (str, optional): Label for the plot (used when scatter is True).
    - highlight_best (bool): If True, highlight the point with the highest value.
    - perc (bool): If True, format y-axis values as percentages.
    - save (bool): If True, save the plot as an image file.
    - return_ax (bool): If True, return the matplotlib Axes object.

    Returns:
    Optional[plt.Axes]: The matplotlib Axes object if return_ax is True, otherwise None.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(size)
    if scatter:
        ax.plot(labels, values,color=color, linewidth=2,marker="o", label=label)
    else:
        ax.plot(labels, values,color=color, linewidth=2)
    if highlight_best:
        ax.scatter(values.index(max(values)), max(values), color="springgreen", label="best")
    ax.set_title(title, weight="bold")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="-", axis="y")
    ax.tick_params(length=0, axis="y")
    ax.set_xticks(labels)
    ax.set_ylabel(ylabel, fontname='Arial', fontweight="bold", size=12)
    ax.set_xlabel(xlabel, fontname='Arial', fontweight="bold", size=12)
    ax.spines['left'].set_position(('data', 0))
    if scatter:
        ax.set_ylim(min(values), max(values)+np.mean(values)*0.1)
    else:
        ax.set_ylim(min(values), max(values))
    if perc:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    if highlight_best:
        ax.legend()
    if return_ax:
        return ax
    if save:
        if title is None:
            title = "no_title"
        title=retitle(title)
        plt.savefig(title, bbox_inches='tight')
    else:
        plt.show()

    if save:
        title=retitle(title)
        if title_suffix is not None:
            title+=title_suffix
        plt.savefig(title+".png",bbox_inches='tight')
    else:
        plt.show()

def retitle(title: Optional[str], csv: bool = False) -> str:
    """
    Format and check the uniqueness of a title for saving files.

    Parameters:
    - title (str, optional): Title to be formatted.
    - csv (bool): If True, check for uniqueness with ".csv" extension.

    Returns:
    str: Formatted and unique title.
    """

    if title is None:
        title = "no_title"
    title=title.lower().replace(" ", "_")
    title=title.replace(".", "_")
    if title+".png" in os.listdir():
        idx=1 
        while title+"_"+str(idx)+".png" in os.listdir():
            idx+=1
        title=title+"_"+str(idx)
    return title

def deliveries_or_cr(deliveries: List[int], open_rates: List[float], click_rates: List[float], indices: List[str],
                     annotate: bool = False, or_color: str = "black", ylabel1: str = "Unique recipients",
                     ylabel2: str = "Open/Click Rate", cr_color: str = "red", xlabel: str = "Hour of the Day",
                     deliveries_color: str = "royalblue", title: str = "", title_suffix: Optional[str] = None,
                     rotate_labels: bool = False, size: tuple = (10, 7), save: bool = False) -> None:
    """
    Create a bar plot with two y-axes representing deliveries and open/click rates.

    Parameters:
    - deliveries (List[int]): List of delivery values.
    - open_rates (List[float]): List of open rates.
    - click_rates (List[float]): List of click rates.
    - indices (List[str]): List of indices for the x-axis.
    - annotate (bool): If True, add value labels to the bar plot.
    - or_color (str): Color of the open rate line plot.
    - ylabel1 (str): Label for the left y-axis.
    - ylabel2 (str): Label for the right y-axis.
    - cr_color (str): Color of the click rate line plot.
    - xlabel (str): Label for the x-axis.
    - deliveries_color (str): Color of the deliveries bar plot.
    - title (str): Title for the plot.
    - title_suffix (str, optional): Suffix to be added to the title.
    - rotate_labels (bool): If True, rotate x-axis labels.
    - size (tuple): Size of the figure. Default is (10, 7).
    - save (bool): If True, save the plot as an image file.

    Returns:
    None
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(size)
    ax.bar(range(len(deliveries)), deliveries.tolist(), color=deliveries_color)
    if annotate:
        add_value_labels(ax)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices)
    ax.grid(linestyle="--", axis="y")
    ax.tick_params(length=0, axis="x")
    ax1=ax.twinx()
    for i in ax1.spines:
        ax1.spines[i].set_visible(False)
    ax1.plot(open_rates.tolist(), color=or_color, linewidth=4, label="Open Rate",alpha=0.7)
    ax1.plot(click_rates.tolist(), color=cr_color, linewidth=4, label="Click To Open Rate",alpha=0.7)
    ax.set_ylabel(ylabel1, size=12, color=deliveries_color, weight="bold")
    ax1.set_ylabel(ylabel2, size=12, weight="bold")
    ax.tick_params(axis='y', length=0,color=deliveries_color, labelsize=12,labelcolor=deliveries_color)
    ax1.tick_params(axis='y', length=0, labelsize=12)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel(xlabel, size=12, weight="bold")
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    if rotate_labels:
        ax.set_xticklabels(indices,rotation=45)
    ax1.legend()
    if save:
        title=retitle(title)
        if title_suffix is not None:
            title+=title_suffix
        plt.savefig(title+".png",bbox_inches='tight')
    else:
        plt.show()

def list_files(directory: str = "Data", limit: int = 0) -> List[str]:
    """
    List files in a directory sorted by modification time in descending order.

    Parameters:
    - directory (str, optional): Directory path. Default is "Data".
    - limit (int, optional): Limit the number of files to return. Default is 0 (no limit).

    Returns:
    List[str]: List of file names sorted by modification time.
    """
    files=[str(i) for i in sorted(Path(directory).iterdir(), key=os.path.getmtime, reverse=True)]
    if limit>0:
        return files[:limit]
    else:
        return files

def n_required(p1: float, p2: float) -> int:
    """
    Calculate the required sample size for a two-sample proportion test.

    Parameters:
    - p1 (float): Proportion for the first sample.
    - p2 (float): Proportion for the second sample.

    Returns:
    int: Required sample size.
    """    
    za=1.96
    zb=0.842
    x=p2-p1
    n=(za+zb)**2
    n=n*(p1*(1-p1)+p2*(1-p2))
    n=n/x**2
    return round(np.ceil(n))

def calculate_p2(n: int, p1: float) -> float:
    """
    Calculate the value of p2 for a given sample size n and p1.

    Parameters:
    - n (int): Sample size.
    - p1 (float): Proportion for the first sample.

    Returns:
    float: Calculated value of p2.
    """    
    p2=2
    too_small=False
    if n_required(p1,p2) > n:
        return None
    while too_small == False:
        required=n_required(p1,p2-0.001)
        if required <= n:
            p2=p2-0.001
        else:
            too_small=True
    return round(p2, 4)

def ford_campaign_type(df: pd.DataFrame, delivery_label: str = "label_delivery") -> pd.DataFrame:
    """
    Determine the campaign type based on the delivery label and add a 'campaign_type' column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - delivery_label (str, optional): Column name representing the delivery label. Default is "label_delivery".

    Returns:
    pd.DataFrame: DataFrame with an added 'campaign_type' column.
    """
    df["campaign_type"]=df[delivery_label].str.startswith("W").map({True:"adhoc", False:np.NaN}).combine_first(df[delivery_label].str.lower().str.startswith("program").map({True:"program", False:np.NaN}).combine_first(df[delivery_label].str.lower().str.startswith("recurring").map({True:"program", False:np.NaN})))
    return df

def scatter_plot(x: List[float], y: List[float], color: str = "royalblue", lobf: bool = False,
                 xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None,
                 size: Optional[Tuple[int, int]] = None, save: bool = False) -> plt.Axes:
    """
    Create a scatter plot with optional line of best fit.

    Parameters:
    - x (List[float]): List of x-axis values.
    - y (List[float]): List of y-axis values.
    - color (str, optional): Color of the scatter plot points. Default is "royalblue".
    - lobf (bool): If True, plot a line of best fit.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - title (str, optional): Title for the plot.
    - size (Tuple[int, int], optional): Size of the figure. If not provided, calculated based on data ratios.
    - save (bool): If True, save the plot as an image file.

    Returns:
    plt.Axes: Matplotlib Axes object.
    """
    fig, ax = plt.subplots()
    if size is None:
        ratio=max(x)/max(y)
        if ratio>=1:
            y_size=10
            x_size=round(y_size*ratio)
        elif ratio<1:
            x_size=10
            y_size=round(x_size/ratio)
    else:
        x_size=size[0]
        y_size=size[1]
    fig.set_size_inches(x_size, y_size)
    ax.scatter(x, y, color=color)
    x_plot=np.unique(x)
    gradient=np.poly1d(np.polyfit(x, y, 1))
    y_plot=gradient(np.unique(x))
    if lobf:
        ax.plot(x_plot, y_plot, c="Black", linestyle="--", label="Line of Best Fit")
    split_x=max(x_plot)-min(x_plot)
    middle_x=split_x/2
    middle_y=gradient(middle_x)
    t=ax.annotate("Pearson correlation coefficient = "+str(round(pearsonr(x,y)[0],2)),(middle_x, middle_y), alpha=1, fontweight="bold",ha="center")
    t.set_bbox(dict(facecolor='red', alpha=0.6, edgecolor='red'))
    ax=set_appearance(ax, [],xlabel, ylabel, title, use_labels=False)
    return ax

def boxplot(df: pd.DataFrame, x: str, y: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
            split: Optional[str] = None, size: Tuple[int, int] = (18, 10), outliers: bool = False,
            title: str = "") -> plt.Axes:
    """
    Create a boxplot with seaborn.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x (str): Column name for the x-axis.
    - y (str): Column name for the y-axis.
    - xlabel (str, optional): Label for the x-axis. Default is None.
    - ylabel (str, optional): Label for the y-axis. Default is None.
    - split (str, optional): Column name for grouping data. Default is None.
    - size (Tuple[int, int], optional): Size of the figure. Default is (18, 10).
    - outliers (bool): If True, show outliers in the boxplot. Default is False.
    - title (str): Title for the plot. Default is an empty string.

    Returns:
    plt.Axes: Matplotlib Axes object.
    """
    if xlabel is None:
        xlabel=x
    if ylabel is None:
        ylabel=y
    fig, ax = plt.subplots(figsize=size)
    sns.boxplot(x=x, y=y, hue=split, showfliers = outliers,data=df,palette="Set2")
    for i in ax.spines:
        ax.spines[i].set_color("grey")
    ax.grid(axis="y", linestyle="--")
    ax.set_axisbelow(True)
    plt.xticks(rotation=90)
    ax.tick_params(axis="y",length=0)
    ax.set_ylabel(ylabel, weight="bold")
    ax.set_xlabel(xlabel, weight="bold")
    ax.set_title(title, weight="bold")
    return ax

def deliveries_or_cr_pp(open_rates: List[float], click_rates: List[float], indices: List[str],
                         or_color: str = "black", cr_color: str = "red", title: str = "", save: bool = False) -> None:
    """
    Create a dual-axis plot for open rates and click rates.

    Parameters:
    - open_rates (List[float]): List of open rates.
    - click_rates (List[float]): List of click rates.
    - indices (List[str]): List of indices for the x-axis.
    - or_color (str): Color of the open rate line plot.
    - cr_color (str): Color of the click rate line plot.
    - title (str): Title for the plot.
    - save (bool): If True, save the plot as an image file.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(12,10)
    #ax.set_title(title, weight="bold")
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices)
    ax.grid(linestyle="--", axis="y")
    ax.tick_params(length=0, axis="x")
    ax1=ax.twinx()
    for i in ax1.spines:
        ax1.spines[i].set_visible(False)
    ax.plot(open_rates.tolist(), color=or_color, linewidth=4, label="Open Rate",alpha=0.7)
    ax1.plot(click_rates.tolist(), color=cr_color, linewidth=4, label="Click Rate",alpha=0.7)
    ax.set_ylabel("Percentage female", size=12, color=or_color, weight="bold")
    ax1.set_ylabel("Average age", size=12, color=cr_color,weight="bold")
    ax.tick_params(axis='y', length=0,color=or_color, labelsize=12,labelcolor=or_color)
    ax1.tick_params(axis='y', length=0, color=cr_color, labelsize=12, labelcolor=cr_color)
    ax1.plot([],[], color=or_color, label="Open Rate",linewidth=4)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("Deliveries opened", size=12, weight="bold")
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    ax1.legend()
    ax.set_title(title, fontweight="bold", size=13)
    if save:
        title=retitle(title)
        plt.savefig(title, bbox_inches='tight')
    else:
        plt.show()

def grouped_bar(bar_a: List[float], bar_b: List[float], label_a: str, label_b: str,
                xticks: List[str], color_a: str = "#003478", color_b: str = "#C4D7E5",
                ylabel: Optional[str] = None, xlabel: Optional[str] = None,
                title: Optional[str] = None, return_ax: bool = False, save: bool = False) -> None:
    """
    Create a grouped bar chart showing percentages with values for each of the demographic columns.

    Parameters:
    - bar_a (List[float]): List of values for the first set of bars.
    - bar_b (List[float]): List of values for the second set of bars.
    - label_a (str): Label for the first set of bars.
    - label_b (str): Label for the second set of bars.
    - xticks (List[str]): List of labels for the x-axis.
    - color_a (str): Color for the bars in the first set. Default is "#003478".
    - color_b (str): Color for the bars in the second set. Default is "#C4D7E5".
    - ylabel (str, optional): Label for the y-axis. Default is None.
    - xlabel (str, optional): Label for the x-axis. Default is None.
    - title (str, optional): Title for the plot. Default is None.
    - return_ax (bool): If True, return the Matplotlib Axes object.
    - save (bool): If True, save the plot as an image file.

    Returns:
    None or plt.Axes: If return_ax is True, returns Matplotlib Axes object. Otherwise, returns None.
    """
    #Grouped bar chart showing percentages with values for each of the demographic columns based on TMM alone or all products
    x = np.arange(len(bar_a))  # the label locations
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots()
    fig.set_size_inches(14,11)
    rects1 = ax.bar(x - width/2, bar_a, width, label=label_a, color=color_a)
    rects2 = ax.bar(x+width/2, bar_b, width, label=label_b,color=color_b)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel, size=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=90)
    ax.set_title(title , size=14, weight="bold")
    ax.set_axisbelow(True)
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.grid(linestyle="--", axis="y")
    ax.tick_params(length=0, axis="y")
    ax.legend()
    fig.tight_layout()
    if return_ax:
        return ax
    if save:
        if title is None:
            title = "no_title"
        title=retitle(title)
        plt.savefig(title, bbox_inches='tight')
    else:
        plt.show()

def bar_scatter(bar: List[float], scatter_a: List[float], scatter_b: List[float], label_a: Optional[str] = None,
                label_b: Optional[str] = None, xlabels: Optional[List[str]] = None, y_label: Optional[str] = None,
                y_label1: Optional[str] = None, annotate: bool = False, title: Optional[str] = None,
                title_suffix: Optional[str] = None, save: bool = False) -> None:
    """
    Create a bar and scatter plot with dual y-axes.

    Parameters:
    - bar (List[float]): List of values for the bar plot.
    - scatter_a (List[float]): List of values for the first set of scatter points.
    - scatter_b (List[float]): List of values for the second set of scatter points.
    - label_a (str, optional): Label for the first set of scatter points.
    - label_b (str, optional): Label for the second set of scatter points.
    - xlabels (List[str], optional): List of labels for the x-axis.
    - y_label (str, optional): Label for the left y-axis.
    - y_label1 (str, optional): Label for the right y-axis.
    - annotate (bool): If True, annotate the bar plot with values.
    - title (str, optional): Title for the plot.
    - title_suffix (str, optional): Suffix for the title.
    - save (bool): If True, save the plot as an image file.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(15,10)
    plt.margins(y=0)
    ax.set_xticks(range(len(bar)))
    if xlabels is not None:
        ax.set_xticklabels(xlabels, size=12, weight="bold", rotation=90)
    ax1=ax.twinx()
    ax.bar(range(len(bar)), bar, color="#0276B3")
    ax1.scatter(range(len(bar)), scatter_a, color="#0C1218",  marker="s", s=50, label=label_a)
    ax1.scatter(range(len(bar)), scatter_b, color="#D00C1B", marker="s", s=50, label=label_b)
    ax.set_axisbelow(True)
    ax.grid(linestyle="-")
    ax1.set_ylabel(y_label1, size=14, fontweight="bold")
    ax.set_ylabel(y_label,size=14, fontweight="bold", color="#0276B3")
    ax.set_title(title , size=14, weight="bold")
    for i in ax.spines:
        ax.spines[i].set_visible(False)
        ax1.spines[i].set_visible(False)
    ax.tick_params(length=0, axis="y", colors="#0276B3")
    ax1.tick_params(length=0, axis="y", colors="#D00C1B")
    ax1.legend()
    if annotate:
        for i, value in enumerate(bar):
                    ax.annotate(
                    int(value),                      
                    (i,value),         
                    xytext=(0,5),          
                    textcoords="offset points", 
                    ha='center',                
                    color="#0276B3") 
    if save:
        title=retitle(title)
        if title_suffix is not None:
            title+=title_suffix
        plt.savefig(title+".png",bbox_inches='tight')
    else:
        plt.show()

    #plt.savefig("apple_opens_markets.png",bbox_inches='tight')

def barh_scatter(bar: List[float], scatter_a: List[float], scatter_b: List[float], label_a: Optional[str] = None,
                 label_b: Optional[str] = None, ylabels: Optional[List[str]] = None, x_label: Optional[str] = None,
                 x_label1: Optional[str] = None, title: Optional[str] = None, title_suffix: Optional[str] = None,
                 save: bool = False) -> None:
    """
    Create a horizontal bar and scatter plot with dual x-axes.

    Parameters:
    - bar (List[float]): List of values for the horizontal bar plot.
    - scatter_a (List[float]): List of values for the first set of scatter points.
    - scatter_b (List[float]): List of values for the second set of scatter points.
    - label_a (str, optional): Label for the first set of scatter points.
    - label_b (str, optional): Label for the second set of scatter points.
    - ylabels (List[str], optional): List of labels for the y-axis.
    - x_label (str, optional): Label for the bottom x-axis.
    - x_label1 (str, optional): Label for the top x-axis.
    - title (str, optional): Title for the plot.
    - title_suffix (str, optional): Suffix for the title.
    - save (bool): If True, save the plot as an image file.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10,15)
    plt.margins(y=0)
    ax.set_yticks(range(len(bar)))
    ax.set_yticklabels(ylabels, size=12, weight="bold")
    ax1=ax.twiny()
    ax.barh(range(len(bar)), bar, color="#00A9E0")
    ax1.scatter(scatter_a, range(len(bar)), color="#0C1218",  marker="s", s=50, label=label_b)
    ax1.scatter(scatter_b, range(len(bar)), color="#D00C1B", marker="s", s=50, label=label_a)
    ax.set_axisbelow(True)
    ax.grid(linestyle="-")
    ax1.set_xlabel(x_label1, size=13, fontweight="bold")
    ax.set_xlabel(x_label,size=13, fontweight="bold", color="#00A9E0")
    for i in ax.spines:
        ax.spines[i].set_visible(False)
        ax1.spines[i].set_visible(False)
    ax.tick_params(length=0, axis="x", colors="#0276B3")
    ax1.tick_params(length=0, axis="x", colors="#D00C1B")
    ax1.legend()
    for i, value in enumerate(bar):
                ax.annotate(
                int(value),                      
                (value,i),         
                xytext=(5,0),          
                textcoords="offset points", 
                ha='left',                
                va='center', 
                color="#0276B3") 
    ax.set_title(title , size=14, weight="bold")
    if save:
        title=retitle(title)
        if title_suffix is not None:
            title+=title_suffix
        plt.savefig(title+".png",bbox_inches='tight')
    else:
        plt.show()

def uplift(a: float, b: float, return_string: bool = False) -> float:
    """
    Calculate and print the uplift between two values.

    Parameters:
    - a (float): First value.
    - b (float): Second value.
    - return_string (bool, optional): If True, return the uplift string. Default is False.

    Returns:
    float or str: If return_string is True, returns a string. Otherwise, returns a float.
    """
    if a>b:
        ppts = a-b
        uplift = round((ppts/b)*100,2)
        string = "{0}% (+{1}pts)".format(uplift, round(ppts,2))
        if return_string:
           return string
        print(string)
    elif b>a:
        ppts=b-a
        uplift = round((ppts/a)*100,2)
        string = "{0}% (+{1}pts)".format(uplift, round(ppts,2))
        if return_string:
            return string
        print(string)
    else:
        if return_string:
            return "draw"
        print("draw")
    return(uplift)

def split_to_n(n_desired: int, clicks: List[float], normalised: bool = True) -> List[float]:
    """
    Split a list of clicks into n parts.

    Parameters:
    - n_desired (int): The desired number of parts.
    - clicks (List[float]): List of click values.
    - normalised (bool, optional): If True, normalize the output. Default is True.

    Returns:
    List[float]: List of split click values.
    """
    n_actual = len(clicks)
    total_clicks=sum(clicks)
    desired_output=np.zeros(n_desired)
    current=0
    if n_actual < n_desired:
        start = 0
        end=start+n_actual/n_desired
        while current < n_desired:
            try:
                if int(end)>len(clicks)-1:
                    print("start", start, "end",end, "current",current, "clicks",len(clicks))
                    desired_output[current]=clicks[int(start)]*(int(end)-start)

                elif int(end)>int(start) and end-int(end)>0:                   
                    desired_output[current]=clicks[int(start)]*(int(end)-start)+clicks[int(end)]*(end-int(end))
                else:
                    desired_output[current]+=clicks[int(start)]*(end-start)
                start=end
                end+=n_actual/n_desired
                current+=1
            except:
                print(start, end)
                continue
    elif n_actual > n_desired:
        start = 0
        end=start+n_actual/n_desired        
        while current < n_desired:
            total=0
            start_fract=start-int(start)
            start_index=int(start)
            take_start=1-start_fract
            total+=clicks[start_index]*take_start
            end_fract=end-int(end)
            end_index=int(end)
            take_end=end_fract
            total+=clicks[min(end_index,n_actual-1)]*take_end
            if end_index==n_actual and take_end==0:
                total+=clicks[-1]
            if take_end==1:
                stop_loop=end_index-1
            else:
                stop_loop=min(n_actual-1,end_index)
            for i in range(start_index+1,stop_loop):
                total+=clicks[i]
            start=end
            end+=n_actual/n_desired
            desired_output[current]=total
            current+=1
    else:
        desired_output=np.array(clicks)
    if normalised:
        desired_output=desired_output/desired_output.sum()
    return desired_output
    
def generate_block_counts(blocks: List[int], clicks: List[float]) -> np.ndarray:
    """
    Generate an array of click counts for each block.

    Parameters:
    - blocks (List[int]): List of block values.
    - clicks (List[float]): List of click values.

    Returns:
    np.ndarray: Array of click counts for each block.
    """
    blocks=[int(i) for i in blocks]
    clicks_blocks=dict(zip(blocks, clicks))
    for i in range(1,max(clicks_blocks)):
        if i not in clicks_blocks:
            clicks_blocks[i]=0
    clicks_blocks=dict(sorted(clicks_blocks.items()))
    return np.array(list(clicks_blocks.values()))

def heatmap_plot(clicks_blocks: np.ndarray, color: str = "black", title: Optional[str] = None) -> None:
    """
    Create a heatmap plot of click counts for each block.

    Parameters:
    - clicks_blocks (np.ndarray): Array of click counts for each block.
    - color (str, optional): Color of the bars. Default is "black".
    - title (str, optional): Title of the plot. Default is None.

    Returns:
    None
    """
    print(clicks_blocks)
    fig,ax = plt.subplots()
    fig.set_size_inches(5,5)
    ax.barh(np.arange(clicks_blocks.shape[0]), np.flip(clicks_blocks/clicks_blocks.sum()),color=color)
    ax.set_yticks(np.arange(clicks_blocks.shape[0]))
    ax.grid(axis="x",linestyle="--")
    ax.set_yticklabels(np.flip(np.arange(clicks_blocks.shape[0])))
    ax.set_axisbelow(True)
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylabel("Region of email")
    ax.set_xlabel("Percentage of total email clicks")
    if title==None:
        ax.set_title("Percentage of total clicks made by email recipients per ordered email region", weight="bold")
    else:
        ax.set_title("Percentage of total clicks made by email recipients per ordered email region ({0})".format(title), weight="bold")
    plt.show()

def heatmap(n: int, blocks: List[int], clicks: List[float], to_plot: Optional[bool] = True) -> Optional[np.ndarray]:
    """
    Generate a heatmap of click counts for each block.

    Parameters:
    - n (int): The desired number of parts.
    - blocks (List[int]): List of block values.
    - clicks (List[float]): List of click values.
    - to_plot (bool, optional): If True, plot the heatmap. Default is True.

    Returns:
    Optional[np.ndarray]: If to_plot is False, returns the split array. Otherwise, returns None.
    """
    clicks_blocks=generate_block_counts(blocks,clicks)
    split=split_to_n(n,list(clicks_blocks))
    if to_plot:
        heatmap_plot(split)
    else:
        return split

def tableau_to_sql(script: str) -> str:
    """
    Convert a Tableau calculated field script to SQL syntax.

    Parameters:
    - script (str): Tableau calculated field script.

    Returns:
    str: SQL syntax equivalent to the Tableau script.
    """
    script.replace("IF CONTAINS", "CASE WHEN")
    script=re.sub(" +", " ", script)
    script=script.replace("\n", " ")
    script=script.replace("IF CONTAINS", "CASE WHEN")
    script=script.replace("ELSECASE","")
    script=re.sub("[()]", "", script)
    script=script.replace("\n", " ")
    script=re.sub("\[(.*?)\],", "UPPER(DELIVERY_LABEL) LIKE ",script)
    script=re.sub("\" ", "%' ",script)
    script=re.sub("\"","'%",script)
    script=re.sub("(?<=then ')%","",script)
    script=re.sub("%(?=(')([' ']+)([WHEN|ELSE]))","",script)
    script=re.sub("WHEN", "\nWHEN", script)
    script=script.replace("ELSE", "\nELSE")
    script=script.replace("end", "\nEND")
    print(script)
    return script

def translate_sl(sls: List[str], verbose=False) -> List[str]:
    """
    Takes list of subject lines and returns translations of those subject lines

    Args:
        sls (list of strings): list of subject lines
        verbose (bool, optional): If set to true, it will print percentage progress at 5% intervals. Defaults to False.

    Returns:
        translations (list of strings): list of translated subject lines
    """
    printed_percentages = {}
    translations = []
    translator = translate_v2.Client()
    for idx, sl in enumerate(sls):
        try:
            translations.append(translator.translate(sl)["translatedText"])
        except Exception as e:
            print(f"Error: {e}, Subject line: {sl}")
            translations.append("na")
        percentage = int(idx / len(sls) * 100)
        if verbose:
            if percentage % 5 == 0:
                if percentage not in printed_percentages:
                    printed_percentages[percentage] = True
                    print(f"{percentage}% done")
    return translations

def sl_summary_df(df: pd.DataFrame)->pd.DataFrame:
    """Aggregates KPIs by subject line due to subject lines being reused across deliveries

    Args:
        df (pd.DataFrame): Pandas DataFrame containing following columns: translation (translated subject lines), numbersent, numbertargeted, uniqueclicks, uniqueopens, unsubscribes (KPI integer columns)

    Returns:
        pd.DataFrame: Pandas dataframe containing aggregated KPIs per unique subject line translation
    """
    sl_group_df=df.groupby("translation")[["numbersent","numbertargeted","uniqueclicks","uniqueopens","unsubscribes"]].sum().reset_index()

    sl_group_df=sl_group_df.merge(df[["translation", "tokens"]].drop_duplicates(), on="translation")

    sl_group_df["open_rate"]=sl_group_df.uniqueopens/sl_group_df.numbersent
    
    return sl_group_df
    
def open_rate_plot(df:pd.DataFrame, market:str="All", adjusted:bool=False)->Union[None,plt.axis]:
    """Plots average open rate by week. If market is not all, this will plot open rate by week for given market and include a line to indicate when the open rate seems to have stabilised post-iOS 15 launch.

    Args:
        df (pd.DataFrame): [description]
        market (str, optional): String value in country column to specify market to use for plot. Defaults to "All".
        adjusted (bool, optional): If true, use adjusted open rate instead of raw open rate. Defaults to False.
    """
    if market!="All":
        df=df[df.country==market]
    df.index=range(len(df))
    after_ios=df[df.after_ios].index.min()
    fig,ax=plt.subplots()
    fig.set_size_inches(20,10)
    if adjusted:
        ax.plot(range(len(df)),df.open_rate_adjusted)
    else:
        ax.plot(range(len(df)),df.open_rate)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["wk"].astype(str)+", "+df["mnth"].astype(str)+", "+df["yr"].astype(str),rotation=90)
    ios_y=[i/100 for i in range(0,int(df.open_rate.max()*100)+5,5)]
    ios_x=[after_ios]*len(ios_y)
    ax.plot(ios_x, ios_y, color="red", linestyle="--", label="After iOS 15 laúnch")
    if market != "All":
        stabilisation_index=df[df.stabilisation_point==df.label].index.min()
        stabilisation_x=[stabilisation_index]*len(ios_y)
        ax.plot(stabilisation_x, ios_y, color="green", linestyle="--", label="Stabilisation point")
    set_appearance(ax, use_labels=False)
    ax.legend()
    [ax.spines[i].set_visible(False) for i in ax.spines]
    plt.margins(0.02)
    plt.tight_layout()
    ax.set_title("Open rate by week number ({})".format(market), weight="bold")
    plt.show()
    
    
def n_kmeans(features:np.ndarray, n_lower=2, n_upper=40, return_scores=False)->Union[Dict[int, np.ndarray], Tuple[Dict[int, np.ndarray], List[float]]]:
    """Run kmeans clustering algorithm with number of clusters varying from n_lower to n_upper

    Args:
        features (np.ndarray): Features to use for clustering
        n_lower (int, optional): Lowest number of clusters to use. Defaults to 2.
        n_upper (int, optional): Highest number of clusters to use. Defaults to 40.
        return_scores (bool, optional): If true, return silhouette scores for all numbers of clusters. Defaults to False.

    Returns:
        dict: dictionary containing cluster labels for all numbers of clusters
        list: if return_scores, returns list of silhouette scores for each number of clusters
    """
    scores=[]
    all_labels={}
    for i in range (n_lower,n_upper):
        kmeans=KMeans(n_clusters=i,random_state=1,max_iter=1000)
        kmeans.fit(features)
        clusters=kmeans.predict(features)
        sil_score=silhouette_score(features, clusters)
        scores.append(sil_score)
        all_labels[i]=kmeans.labels_
        print(i, sil_score)
    if return_scores:
        return all_labels, scores
    else:
        return all_labels
    
def stabilisation_point(df:pd.DataFrame)->str:
    """Calculates point at which open rates seemed to stabilise post-iOS 15

    Args:
        df (pd.DataFrame): pandas dataframe containing average open rates per week and date labels

    Returns:
        str: date label indicating stabilisation point
    """
    try:
        slopes=[] #list to store all slopes calculated
        df.index=range(len(df)) 
        for i in df.index.tolist()[:-1]: #for each week in remaining data
            new_filtered_df=df[df.index>i] #filter starting from given week and containing all the remaining weeks
            lr=LinearRegression() #instantiate linear regression
            x=np.array(range(len(new_filtered_df))).reshape(-1,1) #create array of indices for linear regression x variables
            lr.fit(x, new_filtered_df.open_rate.to_numpy()) #select open rates as dependent variables
            slopes.append(lr.coef_) #calculate linear regression slope and append to list of slopes
        for idx, slope in enumerate(slopes): #for each slope
            if idx>0: #if not first element in list
                if slope>slopes[idx-1]: #if the slope is greater than the previous slope then that is where stabilisation happened
                    stabilisation_point=df.iloc[idx]["label"] #key in dictionary is market, value is stabilisation point date label
                    break #stop searching for stabilisation point
    
        return stabilisation_point
    except:
        print(df.country.unique())

def tokenise(texts: List[str]) -> List[str]:
    """
    Tokenises texts for example to remove unneeded punctuation

    Args:
        texts (list of strings): list of translated subject lines

    Returns:
        all_toks (list of strings): list of tokenised subject lines
    """
    nlp = spacy.load("en_core_web_sm")
    all_toks = []
    for text in texts:
        doc = nlp(text)
        toks = []
        for idx, tok in enumerate(doc):
            if (
                tok.text.isalnum()
                or emoji.is_emoji(tok.text)
                or tok.text in ["$", "£", "€", "!", "?"]
                or tok.text.replace(".", "").isalpha()
            ):
                toks.append(tok.text)
            elif bool(re.match("[-]+[0-9]{2}", tok.text)) and doc[idx + 1].text == "%":
                toks.append(tok.text + "%")
            elif "name" in tok.text.lower() and "%" in tok.text.lower():
                toks.append("namevariable")
        all_toks.append(toks)
    return all_toks

def clean_sl(sl: str)->str:
    """
    Cleans given subject line to remove code remnants

    Args:
        sl (string): subject line

    Returns:
        sl (string): cleaned subject line, with code remnants removed
    """
    if type(sl) != str:
        print(sl)
        return str(sl)
    else:
        indices = [s.start() for s in re.finditer('%%', sl)]
        if len(indices)==2:
            to_remove=sl[indices[0]:indices[1]+2]
            if "firstname" in to_remove.lower():
                return sl.replace(sl[indices[0]:indices[1]+2]," namevariable ").strip()
            else:
                return sl.replace(sl[indices[0]:indices[1]+2],"")
        else:
            return(sl)
