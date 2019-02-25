import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os, glob, pandas, re
def plot_confusion(pc, real_y, pred_y, sorted_targetnames, outdir=None,outname=None, cmap=plt.cm.Blues, title=None, plot_numeric=False):
	plt.figure(pc, figsize=(20,20))
	alllabs = set(list(real_y) + list(pred_y))
	order_name = [x for x in sorted_targetnames if x in alllabs]
	
	cm_count=confusion_matrix(real_y, pred_y, labels=order_name)
	cm=cm_count.astype('float')/cm_count.sum(axis=1)[:,np.newaxis]
	
	np.set_printoptions(precision=2)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title + "\nConfusion Matrix")
	plt.colorbar()
	tick_marks=np.arange(len(order_name))
	plt.xticks(tick_marks, order_name, rotation=90)
	plt.yticks(tick_marks, order_name)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	if plot_numeric:
		for i, cas in enumerate(cm_count):
			for j,c in enumerate(cas):
				if c>0:
					plt.text(j-.2, i+.2, float("{0:.3f}".format(c)), fontsize=18)
	if outdir is not None:
		plt.savefig(outdir+"/" + outname + ".png",facecolor='w', dpi=100)
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
