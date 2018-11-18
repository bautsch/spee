import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def monograph3(data, reference_curve=None, num_wells=8, num_trials=5000,
               fcst_range=600, save_dir=None, major=None, scale=None,
               fcst_type=None, plot_range=600, tc_area='unlabeled', show_plot=False,
               prod_time='monthly', percentiles=[10, 50, 90]):

    if major not in ['oil', 'gas', 'boe']:
       print('Please specify either oil, gas, or boe for major.')
       return

    if fcst_type not in ['rate', 'cum']:
        print('Forecast type must be rate or cum.')
        return

    if prod_time not in ['daily', 'monthly']:
        print('Production time must be daily or monthly.')
        return

    if reference_curve is not None:
        if len(reference_curve) < fcst_range:
            if fcst_type == 'cum':
                tail = np.zeros(fcst_range - len(reference_curve))
                tail[:] = reference_curve.max()
                tail_df = pd.DataFrame(data=tail, columns=reference_curve.columns.values)
                frames = [reference_curve, tail_df]
                reference_curve = pd.concat(frames, ignore_index=True)
            else:
                tail = np.zeros(fcst_range - len(reference_curve))
                tail_df = pd.DataFrame(data=tail, columns=reference_curve.columns.values)
                frames = [reference_curve, tail_df]
                reference_curve = pd.concat(frames, ignore_index=True)
    if 'MONTH' in data.columns.values:
        data = data.drop('MONTH', axis=1)
    if 'Row Labels' in data.columns.values:
        data = data.drop('Row Labels', axis=1)

    for i in data.columns.values:
        pos = data.loc[data[i].notnull(), i].idxmax()
        data.loc[(data[i].isnull()) & (data.index > pos), i] = data[i].max()

    if len(data) < plot_range:
        print('Changing plot range to length of dataframe:', plot_range)
        plot_range = len(data)

    if fcst_range < plot_range:
        fcst_range = plot_range

    data[data.isna()] = 0

    if fcst_type == 'rate' and prod_time == 'monthly':
        data /= 30.41667
        if reference_curve is not None:
            reference_curve /= 30.41667
        eurs = data.sum()

    if fcst_type == 'rate' and prod_time == 'daily':
        eurs = data.sum()

    if fcst_type == 'cum':
        data = data/1000
        if reference_curve is not None:
            reference_curve /= 1000
        eurs = data.max()

    fcst_results = np.zeros((num_trials, len(data.iloc[:fcst_range,0])))
    eur_results = np.zeros(num_trials)

    if scale is not None:
        for p in data.columns.values:
            try:
                data[p] *= scale[scale.iloc[:, 0] == p].iloc[:, 1].values[0]
                eurs[p] *= scale[scale.iloc[:, 0] == p].iloc[:, 1].values[0] 
            except:
                print('Could not match scale factor to data. Is the ID the first column in scale?')
                print('propnum:', p)
                continue

    print('Starting simulation.')
    for trial in range(num_trials):
        samples = np.random.choice(data.columns.values, size=num_wells)
        mean_fcst = data[samples][:fcst_range].mean(axis=1)
        mean_eur = eurs[samples].mean()
        fcst_results[trial, :] = mean_fcst
        eur_results[trial] = mean_eur

    fcst_percentiles = np.percentile(fcst_results, percentiles, axis=0)
    eur_percentiles = np.percentile(eur_results, percentiles)

    if fcst_type == 'cum':
        rate_time = np.diff(fcst_percentiles, axis=1)
        if save_dir is not None:
            np.savetxt(save_dir+'/'+tc_area+' Rate-Time.csv', rate_time.transpose(), delimiter=',')
        else:
            np.savetxt(tc_area+' Rate-Time.csv', rate_time.transpose(), delimiter=',')
    else:
        if save_dir is not None:
            np.savetxt(save_dir+'/'+tc_area+' rate-time.csv', fcst_percentiles.transpose(), delimiter=',')
        else:
            np.savetxt(tc_area+' Rate-Time.csv', fcst_percentiles.transpose(), delimiter=',')

    eur_plot_values = sns.kdeplot(eur_results, shade=True, vertical=True).get_lines()[0].get_data()
    plt.close()
    kde_values = [eur_plot_values[0][np.abs(eur_plot_values[1] - eur_percentiles[p]).argmin()] for p in range(len(eur_percentiles))]

    if major == 'oil':
        plt.figure(tc_area+' Oil Forecasts', figsize=(12.41, 5.8))
    elif major =='boe':
        plt.figure(tc_area+' BOE Forecasts', figsize=(12.41, 5.8))
    else:
        plt.figure(tc_area+' Gas Forecasts', figsize=(12.41, 5.8))

    if fcst_type == 'cum':
        fcst_fig = plt.subplot(121)
        fcst_ax = plt.gca()
        plt.grid(True,which="both",ls="-")
        plt.plot(np.linspace(0, plot_range, num=plot_range), fcst_results[:1000, :plot_range].transpose(), 'k', alpha=0.01)
        plt.plot(np.linspace(0, plot_range, num=plot_range), fcst_percentiles[:, :plot_range].transpose(), 'r', alpha=0.75)
        if reference_curve is not None:
            plt.plot(np.linspace(1, plot_range, num=plot_range), reference_curve[:plot_range], alpha=0.75)
        plt.xlim(1, plot_range)
        fcst_ax.set_ylim(bottom=0)
        fcst_y_range = fcst_ax.get_ylim()
        if major == 'oil':
            plt.ylabel('Average Cumulative Oil Production, MBbl')
        elif major =='boe':
            plt.ylabel('Average Cumulative BOE Production, MBOE')
        else:
            plt.ylabel('Average Cumulative Gas Production, MMcf')
        if prod_time == 'monthly':
            plt.xlabel('Months on Production')
        else:
            plt.xlabel('Days on Production')

        plt.subplot(122)
        eur_ax = plt.gca()
        sns.kdeplot(eur_results, shade=True, vertical=True, color='k')

        for idx, p in enumerate(eur_percentiles):
            plt.plot([0, kde_values[idx]], [p, p], color='r', alpha=0.5)
        if major == 'oil':
            plt.ylabel('Oil EUR, MBbl')
        elif major =='boe':
            plt.ylabel('BOE EUR, MBOE')
        else:
            plt.ylabel('Gas EUR, MMcf')
        eur_ax.set_xticklabels([])
        plt.xlabel('Distribution')
        eur_ax.set_ylim(fcst_y_range)
        if major == 'oil':
            if reference_curve is not None:
                curve_text = ''
                for curve in reference_curve.columns:
                    curve_text += curve+' Oil EUR: '+str('{:.0f}'.format(reference_curve[curve].max()))+' MBbl'+'\n'
                plt.figtext(0.7, 0.25, curve_text
                                    +'P'+str(percentiles[0])+' Oil EUR: '+str('{:.0f}'.format(eur_percentiles[2]))+' MBbl'+'\n'
                                    +'P'+str(percentiles[1])+' Oil EUR: '+str('{:.0f}'.format(eur_percentiles[1]))+' MBbl'+'\n'
                                    +'P'+str(percentiles[2])+' Oil EUR: '+str('{:.0f}'.format(eur_percentiles[0]))+' MBbl',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
            else:
                plt.figtext(0.7, 0.25, 'P'+str(percentiles[0])+' Oil EUR: '+str('{:.0f}'.format(eur_percentiles[2]))+' MBbl'+'\n'
                                    +'P'+str(percentiles[1])+' Oil EUR: '+str('{:.0f}'.format(eur_percentiles[1]))+' MBbl'+'\n'
                                    +'P'+str(percentiles[2])+' Oil EUR: '+str('{:.0f}'.format(eur_percentiles[0]))+' MBbl',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
        elif major == 'boe':
            if reference_curve is not None:
                plt.figtext(0.7, 0.25, 'Ref BOE EUR: '+str('{:.0f}'.format(reference_curve.max()))+' MBOE'+'\n'
                                    +'P'+str(percentiles[0])+' BOE EUR: '+str('{:.0f}'.format(eur_percentiles[2]))+' MBOE'+'\n'
                                    +'P'+str(percentiles[1])+' BOE EUR: '+str('{:.0f}'.format(eur_percentiles[1]))+' MBOE'+'\n'
                                    +'P'+str(percentiles[2])+' BOE EUR: '+str('{:.0f}'.format(eur_percentiles[0]))+' MBOE',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
            else:
                plt.figtext(0.7, 0.25, 'P'+str(percentiles[0])+' BOE EUR: '+str('{:.0f}'.format(eur_percentiles[2]))+' MBOE'+'\n'
                                    +'P'+str(percentiles[1])+' BOE EUR: '+str('{:.0f}'.format(eur_percentiles[1]))+' MBOE'+'\n'
                                    +'P'+str(percentiles[2])+' BOE EUR: '+str('{:.0f}'.format(eur_percentiles[0]))+' MBOE',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
        else:
            if reference_curve is not None:
                plt.figtext(0.7, 0.25, 'Ref Gas EUR: '+str('{:.0f}'.format(reference_curve.max()))+' MMcf'+'\n'
                                    +'P'+str(percentiles[0])+' Gas EUR: '+str('{:.0f}'.format(eur_percentiles[2]))+' MMcf'+'\n'
                                    +'P'+str(percentiles[1])+' Gas EUR: '+str('{:.0f}'.format(eur_percentiles[1]))+' MMcf'+'\n'
                                    +'P'+str(percentiles[2])+' Gas EUR: '+str('{:.0f}'.format(eur_percentiles[0]))+' MMcf',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
            else:
                plt.figtext(0.7, 0.25, 'P'+str(percentiles[0])+' Gas EUR: '+str('{:.0f}'.format(eur_percentiles[2]))+' MMcf'+'\n'
                                    +'P'+str(percentiles[1])+' Gas EUR: '+str('{:.0f}'.format(eur_percentiles[1]))+' MMcf'+'\n'
                                    +'P'+str(percentiles[2])+' Gas EUR: '+str('{:.0f}'.format(eur_percentiles[0]))+' MMcf',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

        if save_dir is not None:
            plt.savefig(save_dir+'/'+tc_area+' Forecast and EUR Distribution')
        else:
            plt.savefig(tc_area+' Forecast and EUR Distribution')

        if show_plot:
            plt.show()
        plt.close()

    elif fcst_type == 'rate':
        fcst_fig = plt.subplot(111)
        fcst_ax = plt.gca()
        plt.grid(True,which="both",ls="-")
        plt.plot(np.linspace(0, plot_range, num=plot_range), fcst_results[:100, :plot_range].transpose(), 'k', alpha=0.1)
        plt.plot(np.linspace(0, plot_range, num=plot_range), fcst_percentiles[:, :plot_range].transpose(), 'r', alpha=0.75)
        if reference_curve is not None:
            for curve in reference_curve.columns.values:
                plt.plot(np.linspace(1, plot_range, num=plot_range), reference_curve.loc[:plot_range-1, curve], alpha=0.75)
        plt.xlim(1, plot_range)
        if prod_time == 'monthly':
            plt.xticks(np.arange(1, plot_range, 12))
            fcst_ax.set_ylim(bottom=1)
        else:
            plt.xticks(np.arange(1, plot_range, 30))
            if major == 'oil':
                fcst_ax.set_ylim(bottom=100)
            else:
                fcst_ax.set_ylim(bottom=100)
        fcst_fig.set_yscale('log')
        if major == 'oil':
            plt.ylabel('Average Oil Production, Bbl/day')
        elif major =='boe':
            plt.ylabel('Average BOE Production, BOE/day')
        else:
            plt.ylabel('Average Gas Production, MMcf/day')
        if prod_time == 'monthly':
            plt.xlabel('Months on Production')
        else:
            plt.xlabel('Days on Production')

        if save_dir is not None:
            plt.savefig(save_dir+'/'+tc_area+' Forecast')
        else:
            plt.savefig(tc_area+' Forecast')

        if show_plot:
            plt.show()
        plt.close()

