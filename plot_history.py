import pandas as pd
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import argparse


def plot_history(input_path):
    df = pd.read_hdf(input_path, "history")
    input_dir = os.path.dirname(input_path)
    input_hist_file = os.path.basename(input_path)

    plt.plot(df["loss"], label="loss (age)")
    plt.plot(df["val_loss"], label="val_loss (age)")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(input_dir, input_hist_file[8:-5]+"_loss.png"))
    plt.cla()
    
    plt.plot(df["mean_average_error"], label="mean_average_error (age)")
    plt.plot(df["val_mean_average_error"], label="val_mean_average_error (age)")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Average Error")
    plt.legend()
    plt.savefig(os.path.join(input_dir, input_hist_file[8:-5]+"_accuracy.png"))