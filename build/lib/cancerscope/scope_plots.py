import matplotlib
matplotlib.use('Agg')
from plotnine import *
from pandas.api.types import CategoricalDtype
from config import REF_LABELCOUNTS, REF_DISEASECODES
import glob
import re
import pandas as pd
import numpy as np
import os, sys

#### Get class sizes  
labelsize_df = pd.read_csv(REF_LABELCOUNTS, delimiter="\t", header=None)
labelsize_df.columns = ["classlabel", "class_size"]
labelsize_df["class_size_rescaled"] = labelsize_df["class_size"].astype(float)/labelsize_df["class_size"].astype(float).max()
labelsize_df['label_sorted'] = [re.split('_[T,N]S$', m)[0] for m in  labelsize_df.classlabel.tolist()]
labelsize_df['tissuestatus'] = ["Tumor" if m is True else "Normal" for m in ['_TS' in m for m in  labelsize_df.classlabel.tolist()]]
labelsize_df['pred'] = labelsize_df['class_size_rescaled']
labelsize_df['Source'] = None

#### Get predictions across all cases, categories, models 
codes_disease = pd.read_csv(REF_DISEASECODES, delimiter="\t")

#### Function to make plot-ready dataframe from ensemble prediction dictionary (each model = each key, respective value is a 3d tensor)
def get_plotting_df(dict_with_preds, x_sample_names=None):
        modelnames = dict_with_preds.keys()
        df_temp_matrix = np.array([dict_with_preds[i] for i in modelnames])
        num_models, num_samples, num_preds, num_tuples_preds = df_temp_matrix.shape
        flat_matrix = df_temp_matrix.flatten()
        for i in range(0, num_models):
                for j in range(0, num_samples):
                        if i == 0 and j == 0:
                                ret_df = pd.DataFrame(df_temp_matrix[i, j]); ret_df["model"] = modelnames[i]; ret_df["sample_ix"] = j
                        else:
                                new_data_line = pd.DataFrame(df_temp_matrix[i, j]); new_data_line["model"] = modelnames[i]; new_data_line["sample_ix"] = j
                                ret_df = ret_df.append(new_data_line)
        ret_df.columns = ["classlabel", "pred", "model", "sample_ix"]
        if x_sample_names is not None:
                ret_df['sample_name'] = [x_sample_names[m] for m in ret_df['sample_ix'].tolist()]
        ret_df['tissuestatus'] = ["Tumor" if m is True else "Normal" for m in ['_TS' in m for m in  ret_df.classlabel.tolist()]]
        ret_df['label'] = [re.split('_[T,N]S$', m)[0] for m in  ret_df.classlabel.tolist()]
        return ret_df


#### Function to actually plot!  
def plot_cases(ensembledf_with_samples, plot_df_with_samples, outdir, save_txt=False):
	### ensembledf_with_samples is received as output from scope_ensemble.get_ensemble_score()
	### plot_df_with_samples is received as output from scope_plots.get_plotting_df()
	### Needs global dfs labelsize_df, codes_disease
	global labelsize_df, codes_disease
	## First check if sample names provided, otherwise use sample indices as names
	if 'sample_name' not in ensembledf_with_samples:
		sys.stdout.write("\nNo sample names provided, using sample indices\n")
		ensembledf_with_samples['sample_name'] = ensembledf_with_samples['sample_ix']
		plot_df_with_samples['sample_name'] = plot_df_with_samples['sample_ix']
	elif 'sample_name' not in plot_df_with_samples:
		sys.stdout.write("\nNo sample names provided, using sample indices\n")
		ensembledf_with_samples['sample_name'] = ensembledf_with_samples['sample_ix']
		plot_df_with_samples['sample_name'] = plot_df_with_samples['sample_ix']
	#### Get top ranked predictions for each sample
	ensembledf_with_samples_rank1 = ensembledf_with_samples[ensembledf_with_samples.rank_pred ==1]
	ensembledf_with_samples_rank1 ['label_sorted'] = [re.split('_[T,N]S$', m)[0] for m in ensembledf_with_samples_rank1.label.tolist()]
	ensembledf_with_samples_rank1 ['tissuestatus'] = ["Tumor" if m is True else "Normal" for m in ['_TS' in m for m in ensembledf_with_samples_rank1.label.tolist()]]
	#### Prepare merged plot df
	plot_df_with_samples['pred'] = plot_df_with_samples.pred.astype(float)
	df_plt = pd.merge(plot_df_with_samples, codes_disease, how='left', left_on='label', right_on='Code Name')
	df_plt = df_plt.sort_values(['Source', 'classlabel'], ascending=[1,0]).reset_index()
	label_list = df_plt['label'].unique().tolist()
	label_cat = CategoricalDtype(categories = label_list, ordered=True)
	df_plt['label_sorted'] = df_plt['label'].astype(str).astype(label_cat)
	#### Now plot for each sample
	for s in df_plt.sample_name.unique().tolist():
		df2 = df_plt[df_plt.sample_name==s]
		g1 = (ggplot(df2, aes('label_sorted', 'pred', colour='factor(Source)')) + geom_bar(data=labelsize_df, alpha=0.3, colour='white', stat='identity') + geom_point(alpha=0.5, size=0.8, colour="grey") + theme_bw(base_size=12) + facet_grid("tissuestatus~sample_name") + stat_summary() + theme(axis_text_x=element_text(rotation=45, size=8, hjust=1), legend_position=(.5, -.15), legend_direction='horizontal')+ labs(x="Category", y="Confidence (scaled)", color='Organ system') + ggtitle("Grey bars indicate fraction of training samples relative to biggest class\nBlack arrow indicates top-voted class and mean(score) from contributing models") + geom_point(data=ensembledf_with_samples_rank1[ensembledf_with_samples_rank1.sample_name==s], shape=5, colour="black"));
		g1.save("/".join([outdir, "SCOPE_sample-" + str(s) + "_predictions.svg"]), height=6, width=14, units='in', dpi='500')
	if save_txt is True:
		df_plt.to_csv(outdir + "/SCOPE_allPredictions.txt", sep="\t", index=False)
		ensembledf_with_samples.to_csv(outdir + "/SCOPE_topPredictions.txt", sep="\t", index=False)
	return

