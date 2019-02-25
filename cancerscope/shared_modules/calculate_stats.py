import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy
def write_stats(outdir, pc, outdict,keyset, outname,title):
        #Sort by key alphabetical order
        valueset = []
        #keyset = sorted(outdict.keys())
        for key in keyset:
                if key in outdict:
                        valueset.append(outdict[key])

        with open(outdir + "/" + outname + ".txt", 'w') as f_out:
                for key in keyset:
			if key in outdict: 
                        	f_out.write('{0}\t{1}\n'.format(key,outdict[key]))

	#Plot sorted set
        pyplot.figure(pc)
        index = numpy.arange(len(outdict))
        pyplot.xticks(rotation=90)
	barwidth=1; pyplot.bar(range(len(valueset)), valueset, barwidth, alpha=0.8, align='center')
        pyplot.grid(); pyplot.title(title); pyplot.xticks(index, keyset, size='small')
        pyplot.tick_params(axis='both', which='major', labelsize=5.5)
        pyplot.ylabel(outname); pyplot.ylim(1e-4,1); pyplot.savefig(outdir+"/"+outname+".png",facecolor='w')

def calculate_precision(true_labs, pred_labs):
	test_classes = set(list(pred_labs) + list(true_labs))	
	#test_classes = set(pred_labs + true_labs)
        correct_counts = {c: 0 for c in test_classes}
        total_counts = {c: 0 for c in test_classes}
        precision = {c: 0 for c in test_classes}
        for sample in xrange(len(true_labs)):
                actual_class = str(true_labs[sample])
                predicted_class = str(pred_labs[sample])
                if actual_class == predicted_class:
                        correct_counts[actual_class] += 1
                        total_counts[actual_class] += 1
                else:
                        total_counts[predicted_class] += 1

        for c in correct_counts:
                precision[c] = (correct_counts[c] * 1.0) / max(1.0, total_counts[c] * 1.0)

        return(precision)

def calculate_recall(true_labs, pred_labs):
	test_classes = set(list(pred_labs) + list(true_labs))
        #test_classes = pred_labs + true_labs
        correct_counts = {c: 0 for c in test_classes}
        total_counts = {c: 0 for c in test_classes}
        recall = {c: 0 for c in test_classes}
        for sample in xrange(len(true_labs)):
                actual_class = str(true_labs[sample])
                predicted_class = str(pred_labs[sample])
                if actual_class == predicted_class:
                        correct_counts[actual_class] += 1
                        total_counts[actual_class] += 1
                else:
                        total_counts[actual_class] += 1

        for c in correct_counts:
                recall[c] = (correct_counts[c] * 1.0) / max(1.0, total_counts[c] * 1.0)

        return(recall)

def calculate_fbeta_score(true_labs, precision, recall,pred_labs):
        test_classes = set(list(pred_labs) + list(true_labs))
	#test_classes = pred_labs + true_labs
        correct_counts  = {c: 0 for c in test_classes}
        fbeta = {c: 0 for c in test_classes}
        for c in correct_counts:
                fbeta[c] = 2.0 * ((precision[c] * recall[c]) / max(1.0, (precision[c] + recall[c])))

        return(fbeta)

