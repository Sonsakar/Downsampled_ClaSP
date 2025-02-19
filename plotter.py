import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import numpy as np


def critical_distance(df, title=None, filepath=None):
    avg_rank = df.groupby('orig_TS_ID').score.rank(pct=False, ascending=False).groupby(df.Algorithm).mean()
    p_vals = sp.posthoc_nemenyi_friedman(
        df,
        melted=True,
        block_col='orig_TS_ID',
        group_col='Algorithm',
        y_col='score',
    )

    fig = plt.figure(figsize=(10, 2), dpi=100)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = "12"
    font = {'family':'arial','color':'black','size':12}
    plt.title(title, loc = 'left', fontdict=font, y=1.2)
    
    sp.critical_difference_diagram(avg_rank, p_vals,
                               label_props={'color': 'black'},
                               elbow_props={'color': 'gray'}                              
                              )
    
    if filepath:
         fig.savefig(filepath, bbox_inches='tight')

    plt.show()
	
	
def plot_clasp(clasp, gt_cps=None, heading=None, ts_name=None, fig_size=(20, 10), font_size=18, file_path=None):

        fig, axes = plt.subplots(clasp.time_series.shape[1] + 1, sharex=True, gridspec_kw={"hspace": .05},
                                 figsize=fig_size)

        ts_axes, profile_ax = axes[:-1], axes[-1]

        if gt_cps is not None:
            segments = [0] + gt_cps.tolist() + [clasp.time_series.shape[0]]
            for dim, ax in enumerate(ts_axes):
                for idx in np.arange(0, len(segments) - 1):
                    ax.plot(np.arange(segments[idx], segments[idx + 1]),
                            clasp.time_series[segments[idx]:segments[idx + 1], dim])

            profile_ax.plot(np.arange(clasp.profile.shape[0]), clasp.profile, color="black")
        else:
            for dim, ax in enumerate(ts_axes):
                ax.plot(np.arange(clasp.time_series.shape[0]), clasp.time_series[:, dim])

            profile_ax.plot(np.arange(clasp.profile.shape[0]), clasp.profile, color="black")

        if heading is not None:
            axes[0].set_title(heading, fontsize=font_size)

        if ts_name is not None:
            axes[0].set_ylabel(ts_name, fontsize=font_size)

        profile_ax.set_xlabel("Split Point", fontsize=font_size)
        profile_ax.set_ylabel("ClaSP Score", fontsize=font_size)

        for ax in axes:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)


        for idx, found_cp in enumerate(clasp.change_points):
            axes[1].axvline(x=found_cp, linewidth=2, color="r", label="Predicted Change Point" if idx == 0 else None)

        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")

        return axes
	
	
