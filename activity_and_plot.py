import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ActivityPlotter:
    
    activity_dict = {   
        99: "non-study activity",
        77: "clapping",
        4: "driving",
        3: "ascending stairs",
        2: "descending stairs",
        1: "walking"
    }

    def __init__(self,dataframe):
        self.df = dataframe

    def separate_activity(self,activity):
        return self.df[self.df['activity'] == activity]
    
    def make_bar_plot(self):
        fig, ax = plt.subplots()
        plt.xlabel('Activity Type')
        plt.ylabel('Training examples')
        bar_colors = ['tab:blue', 'tab:red', 'tab:brown', 'tab:orange','tab:pink','tab:green']
        self.df['activity'].value_counts().plot(kind='bar',color=bar_colors,title='Training examples by Activity Types')
        plt.show()

    def _axis_plot(self,ax, x, y, title):
        ax.plot(x, y, 'r')
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_xlim([min(x), max(x)])
        ax.grid(True)
    
    def axis_plot(self):
        for activity in self.df['activity'].unique():
            if activity == 99 or activity == 77 or activity ==4:
                continue
            limit = self.df[self.df['activity'] == activity][:180]
            fig, (ax0,ax1) = plt.subplots(nrows=2, sharex=True)
            self._axis_plot(ax0, limit['timestamp'], limit['lw_x'], 'x-axis')
            self._axis_plot(ax1, limit['timestamp'], limit['lw_y'], 'y-axis')
            plt.subplots_adjust(hspace=0.2)
            fig.suptitle(self.activity_dict[activity])
            plt.subplots_adjust(top=0.8)
            plt.show()

    def make_x_y_z_plot(self,df):
        option_colors = sns.color_palette()
        colors =[option_colors[0],option_colors[1],option_colors[2]]
        # create larger subplots
            # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        for i, (ax, variables) in enumerate(zip(axs.flat, [('lw_x', 'lw_y', 'lw_z'), ('lh_x', 'lh_y', 'lh_z'), ('la_x', 'la_y', 'la_z'), ('ra_x', 'ra_y', 'ra_z')])):
            for variable, color in zip(variables, colors):
                sns.lineplot(data=df, x='time_s', y=variable, ax=ax, linewidth=0.5, color=color, label=variable)
            
            ax.legend(loc='upper right', fontsize=12)
            ax.set_title(' - '.join(variables), fontsize=16)
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            ax.grid(True)

        plt.tight_layout()
        plt.show()