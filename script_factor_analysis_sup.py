import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from script_create_plots_sup import model_type_dict, vae_type_dict, vae_param_dict, dark_colors
from src.utils.plotting import create_box_plot


def main():
    # create plots folder if it doesn't exist
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(cur_dir, "plots")):
        os.mkdir(os.path.join(cur_dir, "plots"))

    for m in model_type_dict.keys():
        data = []
        vae_list = []
        for v in vae_type_dict.keys():
            if v == 'Factor-VAE' or v == 'Annealed-VAE':
                continue
            vae_list.append(v)
            fp_function = vae_type_dict[v]
            a = vae_param_dict[v]
            for p in a:
                results_fp = fp_function(m, p)
                with open(results_fp, 'r') as infile:
                    results_dict = json.load(infile)
                f_dict = results_dict['mig_factors']
                m_list = []
                c_list = []
                temp_list = []
                num_factors = 0
                for f in f_dict.keys():
                    m_list.append(f_dict[f])
                    if f == 'arp_chord8':
                        c_list.append('arp_chord4')
                    else:
                        c_list.append(f)
                    num_factors += 1
                if len(m_list) != 0:
                    temp_list.append(m_list)
                    temp_list.append(c_list)
                    temp_list.append(num_factors * [v])
                    data.append(temp_list)
        data = np.concatenate(data, axis=1).T
        df = pd.DataFrame(columns=['MIG', 'Attribute', 'VAE Type'], data=data)
        df['MIG'] = df['MIG'].astype(float)
        y_axis_range = None
        fig, ax = create_box_plot(
            data_frame=df,
            model_list=vae_list,
            d_list=dark_colors,
            x_axis='Attribute',
            y_axis='MIG',
            grouping='VAE Type',
            legend_on=False,
            alpha=0.8,
            stripplot_on=False,
            y_axis_range=y_axis_range
        )
        plt.setp(ax.get_xticklabels(), rotation=45)
        save_path = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'plots', f'sup_factor_analysis_{m}_dmel.pdf'
        )
        plt.tight_layout()
        plt.savefig(save_path)


if __name__ == '__main__':
    main()
