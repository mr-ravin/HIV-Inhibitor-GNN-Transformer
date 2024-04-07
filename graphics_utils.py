import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd 

def save_plot(train_loss_list=[], test_loss_list=[], filter_bucket=1):
        plt.figure()
        red_patch = mpatches.Patch(color='red', label='Train Loss')
        blue_patch = mpatches.Patch(color='blue', label='Test Loss')
        x_axis_data = list(range(1,len(train_loss_list)+1))
        x_axis_data = [x * filter_bucket for x in x_axis_data]
        sns.lineplot(x=x_axis_data, y=train_loss_list, color='red', alpha=0.75)
        sns.lineplot(x=x_axis_data, y=test_loss_list, color='blue', alpha=0.75)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GNN Training Analysis")
        plt.legend(handles=[red_patch, blue_patch], loc='upper right')
        plt.savefig('./result/training_analysis.png')
        print("Saved ./result/training_analysis.png")