import os
import shutil
import pandas as pd
from matplotlib.lines import lineStyles

datasets = ['amazon_boys_girls', 'pinterest', 'amazon_men']  # amazon_boys_girls   pinterest amazon_men
models = ['VBPR', 'AMR']  # VBPR   AMR

if os.path.exists('PLOTS'):
    shutil.rmtree('PLOTS')
os.makedirs('PLOTS')

round_value = 4

for dataset in datasets:
    for model in models:
        if dataset == 'amazon_boys_girls':
            if model == 'VBPR':
                ## VBPR
                attack = pd.read_csv(f'{dataset}/{model}/attack-lr0.0001-bs32-18_09_2021_08_26_59.tsv', sep='\t')
                top_20_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_20_relthreshold_1_2021_09_20_08_22_40.tsv',
                    sep='\t')
                top_50_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_50_relthreshold_1_2021_09_20_08_22_40.tsv',
                    sep='\t')
                denoised_performance = pd.read_csv(
                    f'{dataset}/{model}/denoised-overall-performance-lr0.0001-bs32-18_09_2021_08_26_59.tsv',
                    sep='\t')
            elif model == 'AMR':
                attack = pd.read_csv(f'{dataset}/{model}/attack-lr0.0001-bs32-24_09_2021_10_59_36.tsv', sep='\t')
                top_20_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_20_relthreshold_1_2021_09_24_13_24_46.tsv',
                    sep='\t')
                top_50_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_50_relthreshold_1_2021_09_24_13_24_46.tsv',
                    sep='\t')
                denoised_performance = pd.read_csv(
                    f'{dataset}/{model}/denoised-overall-performance-lr0.0001-bs32-24_09_2021_10_59_36.tsv', sep='\t')
        elif dataset == 'pinterest':
            if model == 'VBPR':
                attack = pd.read_csv(f'{dataset}/{model}/attack-vbpr-lr0.0001-bs16-11_10_2021_09_53_09.tsv', sep='\t')
                top_20_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_20_relthreshold_1_2021_10_12_01_13_56.tsv',
                    sep='\t')
                top_50_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_50_relthreshold_1_2021_10_12_01_13_56.tsv',
                    sep='\t')
                denoised_performance = pd.read_csv(
                    f'{dataset}/{model}/denoised-vbpr-overall-performance-lr0.0001-bs16-11_10_2021_09_53_09.tsv',
                    sep='\t')
            elif model == 'AMR':
                # AMR
                attack = pd.read_csv(f'{dataset}/{model}/attack-amr-lr0.0001-bs16-11_10_2021_10_03_24.tsv', sep='\t')
                top_20_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_20_relthreshold_1_2021_10_12_05_42_52.tsv', sep='\t')
                top_50_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_50_relthreshold_1_2021_10_12_05_42_52.tsv', sep='\t')
                denoised_performance = pd.read_csv(
                    f'{dataset}/{model}/denoised-amr-overall-performance-lr0.0001-bs16-11_10_2021_10_03_24.tsv',
                    sep='\t')
        elif dataset == 'amazon_men':
            if model == 'VBPR':
                attack = pd.read_csv(f'{dataset}/{model}/attack-vbpr-lr0.0001-bs16-11_10_2021_06_30_10.tsv', sep='\t')
                top_20_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_20_relthreshold_1_2021_10_11_17_51_33.tsv',
                    sep='\t')
                top_50_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_50_relthreshold_1_2021_10_11_17_51_33.tsv',
                    sep='\t')
                denoised_performance = pd.read_csv(
                    f'{dataset}/{model}/denoised-vbpr-overall-performance-lr0.0001-bs16-11_10_2021_06_30_10.tsv',
                    sep='\t')
            elif model == 'AMR':
                # AMR
                attack = pd.read_csv(f'{dataset}/{model}/attack-amr-lr0.0001-bs16-08_10_2021_20_29_44.tsv', sep='\t')
                top_20_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_20_relthreshold_1_2021_10_11_22_59_59.tsv', sep='\t')
                top_50_clean_performance = pd.read_csv(
                    f'{dataset}/{model}/rec_cutoff_50_relthreshold_1_2021_10_11_22_59_59.tsv', sep='\t')
                denoised_performance = pd.read_csv(
                    f'{dataset}/{model}/denoised-amr-overall-performance-lr0.0001-bs16-08_10_2021_20_29_44.tsv',
                    sep='\t')

        print(f'DATASET:\t{dataset}\tMODEL:\t{model}\n')

        # Columns
        # 'Defense', 'Item', 'Popularity', 'Attack', 'Eps', 'Steps', 'HRBefAtt@50', 'HRAftAtt@50', 'NumImprov', 'PredShift'

        # TABLE 1. Prediction Shift and Number of Improvements
        avg_results = attack.groupby(['Defense', 'Attack', 'Eps', 'Steps'], as_index=False)[
            ['HRBefAtt@50', 'HRAftAtt@50', 'NumImprov', 'PredShift']].mean()
        avg_results = avg_results[(avg_results['Steps'] == 1) & (avg_results['Eps'] == 4)]
        denoised_rows = avg_results[(avg_results['Defense'] == 'denoiser')].reset_index()

        print('Table 1. Prediction Shift and Number of Improvements Performance\n')
        print('Attack\tEps\tSteps\tPS-NoDen\tPS-Den')
        for index, row in avg_results[avg_results['Defense'] == 'nodefense'].iterrows():
            denoised_row = \
            denoised_rows[(denoised_rows['Attack'] == row['Attack']) & (denoised_rows['Eps'] == row['Eps']) & (
                    denoised_rows['Steps'] == row['Steps'])].iloc[0]
            print(
                f"{row['Attack']}\t{row['Eps']}\t{row['Steps']}\t{round(row['PredShift'], round_value)}\t{round(denoised_row['PredShift'], round_value)}")
        print('\n')

        # Table 2. Top-K Stability
        avg_results['DeltaHR@50'] = (avg_results['HRAftAtt@50'] - avg_results['HRBefAtt@50']) / avg_results[
            'HRBefAtt@50']
        denoised_rows = avg_results[(avg_results['Defense'] == 'denoiser')].reset_index()

        print('Table 2. Top-K Stability\n')
        print('Attack\tEps\tSteps\tDHR_NoDen\tDHR_Den\tStability')
        for index, row in avg_results[avg_results['Defense'] == 'nodefense'].iterrows():
            denoised_row = \
            denoised_rows[(denoised_rows['Attack'] == row['Attack']) & (denoised_rows['Eps'] == row['Eps']) & (
                    denoised_rows['Steps'] == row['Steps'])].iloc[0]
            stability = abs(denoised_row['DeltaHR@50'] / row['DeltaHR@50'])
            print(
                f"{row['Attack']}\t{row['Eps']}\t{row['Steps']}\t{round(row['DeltaHR@50'], round_value)}\t{round(denoised_row['DeltaHR@50'], round_value)}\t{round(stability, round_value)}")
        print('\n')

        # Table 3. Top-K Stability
        print('Table 3. Overall Performance\n')

        header = ' '.join([(str(elem) + '\t' + str(elem)) for elem in denoised_performance.columns[1:]])
        print('Top-K\t' + header.replace(' ', '\t'))
        for k in [20, 50]:
            line = f'{k}\t'
            for metric in denoised_performance.columns[1:]:
                den_value = denoised_performance[denoised_performance['k'] == k][metric].values[0]
                if k == 20:
                    line += f'{round(top_20_clean_performance[metric].values[0], round_value)}\t{round(den_value, round_value)}\t'
                elif k == 50:
                    line += f'{round(top_50_clean_performance[metric].values[0], round_value)}\t{round(den_value, round_value)}\t'
            print(line)
        print('\n')


        # Table 3-pos. Top-K Stability
        print('Table 3-pos. Overall Performance\n')

        header = ' '.join([(str(elem) + '\t' + str(elem)) for elem in denoised_performance.columns[1:]])
        print('Defense\t' + header.replace(' ', '\t'))
        line_normal = f'No\t'
        line_denoiser = f'AiD\t'
        line_rv = f'R.V.\t'
        for metric in denoised_performance.columns[1:]:
            for k in [20, 50]:
                if k == 20:
                    den_value = denoised_performance[denoised_performance['k'] == k][metric].values[0]
                    line_normal += f'{round(top_20_clean_performance[metric].values[0], round_value)}\t'
                    line_denoiser += f'{round(den_value, round_value)}\t'
                    line_rv += f'{round((round(den_value, round_value)-round(top_20_clean_performance[metric].values[0], round_value))*100/round(top_20_clean_performance[metric].values[0], round_value), 2)}\t'
                elif k == 50:
                    den_value = denoised_performance[denoised_performance['k'] == k][metric].values[0]
                    line_normal += f'{round(top_50_clean_performance[metric].values[0], round_value)}\t'
                    line_denoiser += f'{round(den_value, round_value)}\t'
                    line_rv += f'{round((round(den_value, round_value)-round(top_50_clean_performance[metric].values[0], round_value))*100/round(top_50_clean_performance[metric].values[0], round_value), 2)}\t'

        print(line_normal)
        print(line_denoiser)
        print(line_rv)
        print('\n')

        print('Figures. Prediction Shift varying epsilon and iterations\n')

        # Figure Change Iterations/Budgets
        import matplotlib.pyplot as plt


        attack = attack.groupby(['Defense', 'Attack', 'Eps', 'Steps'], as_index=False)[
            ['HRBefAtt@50', 'HRAftAtt@50', 'NumImprov', 'PredShift']].mean()
        colors = {4: 'r*', 8: 'b^', 16: 'g.'}
        attacks = {'insa': 'INSA', 'wbsign': 'SIGN'}
        defenses = {'nodefense': 'wo AiD', 'denoiser': 'w AiD'}
        for attack_type in ['insa', 'wbsign']:
            fig = plt.figure()
            ax = plt.subplot(111)
            for defense_type in attack['Defense'].unique():
                for epsilon in attack['Eps'].unique():
                    x = attack['Steps'].unique()
                    y = attack[
                        (attack['Attack'] == attack_type) & (attack['Defense'] == defense_type) & (
                                    attack['Eps'] == epsilon)]
                    y = y.sort_values(by=['Steps'])['PredShift'].to_list()
                    if defense_type == 'nodefense':
                        ax.plot(x, y, f'{colors[epsilon]}-',
                                 label=r'$\epsilon=$' + f'{epsilon} ({defenses[defense_type]})', linewidth=2, markersize=12)
                    else:
                        ax.plot(x, y, f'{colors[epsilon]}--',
                                 label=r'$\epsilon=$' + f'{epsilon} ({defenses[defense_type]})', linewidth=2, markersize=12)
            plt.xlabel(r'$T$', fontsize=18)
            plt.xticks(x, x, fontsize=18)
            plt.ylabel(r'$\mathrm{PS}$', fontsize=18)
            # plt.legend()

            # Put a legend below current axis
            # ax.legend(prop={'size': 12}, loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #           fancybox=True, shadow=True, ncol=3)

            plt.tight_layout()
            # plt.show()
            plt.savefig(f'PLOTS/ps_{dataset}-{model}-{attack_type}.png')

        # least_popular = attack[['Item', 'Popularity']].sort_values(by=['Popularity'])['Item'].unique()[:5]
        # attack = attack[attack['Item'].isin(least_popular)]
