import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp

import numpy as np
import os, glob, pandas, re
def plot_roc(pc, discrete_real_y, y_score, labdict, outdir=None,outname=None, cmap=plt.cm.Blues, title=None, plot_numeric=False):
	#discrete_real_y : Discrete valued class labels (min 0, max = #classes - 1). Dim = (#samples)
	#y_score : not singular prediction, but prediction score for each class for each sample, ordered 0 to #classes - 1. Dim = (#samples, #classes)
	#Binarize input real set
	n_classes = len(set(discrete_real_y))
	if(n_classes != y_score.shape[1]):
		print(n_classes)
		print(y_score.shape[1])
		print("ERROR: Input test range does not correspond to output scores range. Exiting ROC plot")
		return 0 
	
	real_y = label_binarize(discrete_real_y, classes=np.arange(n_classes))
	#Plotting defaults	
	plt.figure(pc, figsize=(20,20))
	np.set_printoptions(precision=2)
	#plt.tight_layout()
	plt.ylabel('TPR')
	plt.xlabel('FPR')
	plt.title(title + '\nReceiver operating characteristic curve')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.0])
	
	#Calculate the fpr and tprs for each label
	fpr=dict(); tpr=dict(); roc_auc=dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(real_y[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	
	fpr["micro"], tpr["micro"], _ = roc_curve(real_y.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		
	#Aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	#Get ROC curves for each class
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	#Calculate AUCs
	mean_tpr /= n_classes
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	
	#Plot data
	plt.plot(fpr["micro"], tpr["micro"], label='micro averaged ROC curve, area {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=":", linewidth=4)
	plt.plot(fpr["macro"], tpr["macro"], label='macro averaged ROC curve, area {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=":", linewidth=4)	
	colors = cycle(['aqua', 'blue', 'cornflowerblue'])
	for i, color in zip(range(n_classes),colors):
		if roc_auc[i] < 0.51:
			plt.plot(fpr[i], tpr[i], color=color, lw=1, label='ROC curve of class {0}, area {1:0.2f}'.format(labdict[i], roc_auc[i]))
	plt.plot([0,1],[0,1], 'k--', lw=2)
	plt.legend(loc='lower right')
	
	if outdir is not None:
		plt.savefig(outdir+"/" + outname + ".png",facecolor='w', dpi=100)
	with open(outdir + "/" + outname  + "_eachclass.txt", 'w') as f_out:
		for key in range(n_classes):
			f_out.write('{0}\t{1}\n'.format(labdict[i], roc_auc[i]))
	
	with open(outdir + "/" + outname  + "_average.txt", 'w') as f_out:
		f_out.write('macro\t{0}'.format(roc_auc["macro"]))
		f_out.write('micro\t{0}'.format(roc_auc["micro"]))
		
	plt.close()

def plot_dir_confusion(indir, pattern=None, merge=False, plot_subs = False):
	#Plot confusion matrix after having generated the entire set of results
	#Finds a file called prediction_labels_*pattern*.txt, and fbeta_*pattern*.txt
	#And outputs a corresponding confusion matrix
	if merge :
		real = []; pred = []
		predfiles = [ indir+"/"+f for f in os.listdir(indir) if re.search(r'(prediction_labels_cv).*\.txt$', f)]
		fbetafile=glob.glob(indir+"/fbeta_full.txt")[0]
		for predfile in predfiles:
			dat = pandas.read_csv(predfile,sep="\t",header=None)
			real.append(dat[[2]][2].values.tolist())
			pred.append(dat[[0]][0].values.tolist())
		pattern = "merged_cv"; real=real[0]; pred=pred[0]
	else:
		if pattern is None:
			print("Input a pattern...")
			return 0	
		else:
			#Single file input
			predfile=glob.glob(indir+"/prediction_labels_"+pattern+".txt")[0]
			fbetafile=glob.glob(indir+"/fbeta_"+pattern+".txt")[0]
			dat = pandas.read_csv(predfile, sep="\t", header=None)
			real = dat[[2]][2].values.tolist()
			pred = dat[[0]][0].values.tolist()
	fscore = pandas.read_csv(fbetafile, sep="\t", header=None)[[0]]
	sorttarget=fscore[0].values.tolist()
	plot_confusion(pc=1, real_y=real, pred_y=pred, sorted_targetnames=sorttarget, outdir=indir, outname="confusion_"+pattern, title="", cmap=plt.cm.viridis)
	plt.close()
	if plot_subs :
		subset_truelabs = ["KICH", "HNSC_TS", "CESC", "ESCA", "THYM", "LUSC"]
		for matchname in subset_truelabs:
			indices = [i for i,x in enumerate(real) if re.search(matchname, x)]
			realx=[real[i] for i in indices]
			predx=[pred[i] for i in indices]
			plot_confusion(pc=1, real_y=realx, pred_y=predx, sorted_targetnames=sorttarget, outdir=indir, outname="confusion_"+pattern+"_" +matchname, title=matchname, cmap=plt.cm.viridis, plot_numeric=True)
			plt.close()	
