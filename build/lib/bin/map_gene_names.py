from __future__ import print_function

def readinput(datfile, header_status, firstcol_ensg=True, transpose_dat=False, sep='\t'):
        "This function takes in a matrix of format samples (columns) x genes (rows) and generates an array of labels (Y) and an array of corresponding input vectors (X). The first column is the hugo ID, unless it is the ensemble ID followed by the hugo ID, in which case firstcol_ensg = True; the 3rd col onwards are the samples"
	print("Input has header? {0}, first two cols are gene names? {1}, setup is sample x gene? {2}, sep is: {3}".format(header_status, firstcol_ensg, transpose_dat, sep))
	if (datfile.endswith('.pkl')):
		with open(datfile, 'r') as infile:
			dataset=cPickle.load(infile)
			X=dataset[0]
			print(dataset[1])
			labels=dataset[1]
			Y=np.asarray(labels)
			samples=dataset[2]
	else:
		header_status = 0 if header_status == True else 'None'
		indat = pd.read_csv(datfile, sep=sep, header = header_status)
		if transpose_dat is True:
			indat2 = np.transpose(indat)
		else:
			indat2 = indat
		samples = list(indat2.columns)
		if firstcol_ensg is True:
			features_ensg = list(indat2.index); 
			features_hugo = list(indat2.iloc[:,0].values)
			sampleix = 1
		else:
			features_ensg = None
			features_hugo = list(indat2.index)
			sampleix = 0
		indat2 = indat2.replace("NA", float('nan'))
		X = indat2.iloc[:,sampleix:].values
		# Final X must be samples(rows) x genes(cols)
		if X.shape[0] == len(features_hugo):
			X = X.transpose()
	return [X, samples, features_ensg, features_hugo]


	## If input FEATURES are not HUGO, map to HUGO from whichever format 
	if in_genecode not in ['HUGO', 'HUGO_ENSG']:
		print("UPDATE: Switching feature mapping to HUGO id")
		genedict = pd.read_csv(genedict_file, sep="\t")
		genedict = genedict.loc[genedict['HUGO'].isin(features_train)]
		genedict = genedict.set_index(in_genecode) # GENCODE, GAF, HUGO
		features_test = list(genedict.ix[features_test]["HUGO"])
	if in_genecode == "HUGO_ENSG":
		features_test = [m.split("_ENSG", 1)[0] for m in features_test]


