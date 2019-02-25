from scipy.stats import pearsonr, spearmanr
import numpy as np
import os 


# #load data
# from io_modules import load_data
# #load in the data
# data_init = load_data('/projects/sjones_prj/tumourchar/data/v8/normed_sets_v8/rastminmax/v8_pogfiltered_12082016_normedrastminmax1_weightednone_keptsmallTrue.pkl.gz')
# #format data
# #test set
# x_test = data_init[2][0]

# #load reconstructed data
# #reconstructed_input = np.load('/home/jeyang/Desktop/DeepLearning/ResultsWithCV/FirstTest/cv5/reconstructions.npy')
# #reconstructed_input = np.load('/home/jeyang/Desktop/DeepLearning/Reconstructions/ChangingHiddenLayerSize/RastMinMax/reconstructedTest_deep_sigmoid_CE_drop0.0_nh_1000_ep_20.npy')
# #reconstructed_input = np.load('/projects/sjones_prj/autoencoder/reconstructedTest_sigmoid_CE_drop0.7_nh_140.npy')

# #load reconstructed data
# #switch out for file of interest
# reconstructed_input_file = '/home/jeyang/Desktop/DeepLearning/Reconstructions/ChangingHiddenLayerSize/RastMinMax/reconstructedTest_deep_sigmoid_CE_drop0.0_nh_1000_ep_20.npy'
# reconstructed_input = np.load(reconstructed_input_file)

# #create directory to output the correlations to
# output_directory_name = '/home/jeyang/Desktop/DeepLearning/Correlations/reconstructedTest_deep_sigmoid_CE_drop0.0_nh_1000_ep_20'
# if not os.path.exists(output_directory_name):
# 	os.makedirs(output_directory_name)

#get_summary(x_test, reconstructed_input)

#------------------------------- Correlation of Means (Genes) --------------------------------
#returns 2 values: pearson_coeff_str, spearman_coeff_str
def corr_of_means_genes(input_actual, input_reconstructed):
	#take means of input and reconstructed input
	#axis 0 is columns
	mean_true = np.mean(input_actual, axis=0)
	#save list of mean_true
	#np.savetxt(output_directory_name + '/mean_true_correlationOfMeans_genes', mean_true)
	mean_predicted = np.mean(input_reconstructed, axis=0)
	#save list of mean_predicted
	#np.savetxt(output_directory_name + '/mean_predicted_correlationofMeans_genes', mean_predicted)

	#Pearson Population Correlation Coefficient
	#assumes normal distribution

	#x and y inputs are each a 1-d array

	pearson_coeff = pearsonr(mean_true, mean_predicted)
	pearson_coeff_str = "pearson: " + str(pearson_coeff[0])
	print(pearson_coeff_str)
	#print("pearson: ", pearson_coeff)


	#Spearman Rank Order Correlation
	#does not assume normal distribution 

	#x and y inputs are each a 1-d array or 2-d array

	spearman_coeff = spearmanr(mean_true, mean_predicted)
	spearman_coeff_str = "spearman: " + str(spearman_coeff[0])
	print(spearman_coeff_str)
	#print("spearman: ", spearman_coeff)

	return pearson_coeff_str, spearman_coeff_str


#------------------------------- Correlation of Genes --------------------------------
#output is pearson_mean_str, pearson_variance_str, spearman_mean_str, spearman_variance_str
def corr_of_genes(input_actual, input_reconstructed):
	size = input_actual.shape[1]
	index = range(0, size)

	#Pearson Population Correlation Coefficient
	#list of correlations of each sample
	sample_correlations_pearson = []

	for i in index:
		pearson_coeff = pearsonr(input_actual[:,i],input_reconstructed[:,i])
		sample_correlations_pearson.append(pearson_coeff[0])

	#save list of sample_correlations_pearson
	#np.savetxt(output_directory_name + '/gene_correlations_pearson', sample_correlations_pearson)

	pearson_mean_genes_str = "pearson_mean: " + str(np.mean(sample_correlations_pearson))
	print(pearson_mean_genes_str)
	#print("pearson_mean: ", np.mean(sample_correlations_pearson))
	pearson_variance_genes_str = "pearson_variance: " + str(np.var(sample_correlations_pearson))
	print(pearson_variance_genes_str)
	#print("pearson_variance: ", np.var(sample_correlations_pearson))

	#Spearman Rank Order Correlation
	#list of correlations of each sample
	sample_correlations_spearman = []

	for i in index:
		spearman_coeff = spearmanr(input_actual[:,i],input_reconstructed[:,i])
		sample_correlations_spearman.append(spearman_coeff[0])

	#save list of sample_correlations_spearman
	#np.savetxt(output_directory_name + '/gene_correlations_spearman', sample_correlations_spearman)

	spearman_mean_genes_str = "spearman_mean: " + str(np.mean(sample_correlations_spearman))
	print(spearman_mean_genes_str)
	#print("spearman_mean: ", np.mean(sample_correlations_spearman))
	spearman_variance_genes_str = "spearman_variance: " + str(np.var(sample_correlations_spearman))
	print(spearman_variance_genes_str)
	#print("spearson_variance: ", np.var(sample_correlations_spearman))

	return pearson_mean_genes_str, pearson_variance_genes_str, spearman_mean_genes_str, spearman_variance_genes_str

#------------------------------- Variance of Input and Output --------------------------------
#output is average_variance_str
def variance_of_genes(input_actual, input_reconstructed):
	#take means of input and reconstructed input
	mean_true = np.mean(input_actual, axis=0)
	mean_predicted = np.mean(input_reconstructed, axis=0)

	size = input_actual.shape[1]
	index = range(0, size)

	#list of variances of each gene
	variance_per_gene = []

	#get variance of each gene
	for j in index:
		difference = abs(mean_predicted[j]-mean_true[j])
		variance_per_gene.append(difference)

	#print variance_per_gene

	average_variance = np.mean(variance_per_gene)
	average_variance_str = "average_var: " + str(average_variance)
	print(average_variance_str)
	#print("average_var: ", average_variance)

	return average_variance_str
#---------------------- Correlation of Means (Samples) ------------------------
#output is pearson_coeff_str_samples, spearman_coeff_str_samples
def corr_of_means_samples(input_actual, input_reconstructed):
	#take means of each sample (average value genes take on)
	#axis 1 is rows
	mean_true = np.mean(input_actual, axis=1)
	#save list of mean_true
	#np.savetxt(output_directory_name + '/mean_true_correlationOfMeans_samples', mean_true)
	mean_predicted = np.mean(input_reconstructed, axis=1)
	#save list of mean_predicted
	#np.savetxt(output_directory_name + '/mean_predicted_correlationOfMeans_samples', mean_predicted)

	#Pearson Population Correlation Coefficient
	#assumes normal distribution

	#x and y inputs are each a 1-d array

	pearson_coeff = pearsonr(mean_true, mean_predicted)
	pearson_coeff_str_samples = "pearson (across samples): " + str(pearson_coeff[0])
	print(pearson_coeff_str_samples)
	#print("pearson: ", pearson_coeff)


	#Spearman Rank Order Correlation
	#does not assume normal distribution 

	#x and y inputs are each a 1-d array or 2-d array

	spearman_coeff = spearmanr(mean_true, mean_predicted)
	spearman_coeff_str_samples = "spearman (across samples): " + str(spearman_coeff[0])
	print(spearman_coeff_str_samples)
	#print("spearman: ", spearman_coeff)

	return pearson_coeff_str_samples, spearman_coeff_str_samples


#------------------------- Correlation of Samples ---------------------------
#output is pearson_mean_str, pearson_variance_str, spearman_mean_str, spearman_variance_str
def corr_of_samples(input_actual, input_reconstructed):
	size = input_actual.shape[0]
	index = range(0, size)

	#Pearson Population Correlation Coefficient
	#list of correlations of each sample
	sample_correlations_pearson = []

	for i in index:
		pearson_coeff = pearsonr(input_actual[i,:],input_reconstructed[i,:])
		sample_correlations_pearson.append(pearson_coeff[0])

	#save list of sample_correlations_pearson
	#np.savetxt(output_directory_name + '/sample_correlations_pearson', sample_correlations_pearson)

	pearson_mean_samples_str = "pearson_mean (across samples): " + str(np.mean(sample_correlations_pearson))
	print(pearson_mean_samples_str)
	#print("pearson_mean: ", np.mean(sample_correlations_pearson))
	pearson_variance_samples_str = "pearson_variance (across samples): " + str(np.var(sample_correlations_pearson))
	print(pearson_variance_samples_str)
	#print("pearson_variance: ", np.var(sample_correlations_pearson))

	#Spearman Rank Order Correlation
	#list of correlations of each sample
	sample_correlations_spearman = []

	for i in index:
		spearman_coeff = spearmanr(input_actual[i,:],input_reconstructed[i,:])
		sample_correlations_spearman.append(spearman_coeff[0])

	#save list of sample_correlations_spearman
	#np.savetxt(output_directory_name + '/sample_correlations_spearman', sample_correlations_spearman)

	spearman_mean_samples_str = "spearman_mean (across samples): " + str(np.mean(sample_correlations_spearman))
	print(spearman_mean_samples_str)
	#print("spearman_mean: ", np.mean(sample_correlations_spearman))
	spearman_variance_samples_str = "spearman_variance (across samples): " + str(np.var(sample_correlations_spearman))
	print(spearman_variance_samples_str)
	#print("spearson_variance: ", np.var(sample_correlations_spearman))

	return pearson_mean_samples_str, pearson_variance_samples_str, spearman_mean_samples_str, spearman_variance_samples_str


#------------------------------ List of Values -------------------------------
#get summary of all correlation values
def get_summary(input_actual, input_reconstructed, output_directory_name):
#save correlations to text file
	pearson_coeff_str, spearman_coeff_str = corr_of_means_genes(input_actual, input_reconstructed)
	pearson_mean_genes_str, pearson_variance_genes_str, spearman_mean_genes_str, spearman_variance_genes_str = corr_of_genes(input_actual, input_reconstructed)
	average_variance_str = variance_of_genes(input_actual, input_reconstructed)
	pearson_coeff_str_samples, spearman_coeff_str_samples = corr_of_means_samples(input_actual, input_reconstructed)
	pearson_mean_samples_str, pearson_variance_samples_str, spearman_mean_samples_str, spearman_variance_samples_str = corr_of_samples(input_actual, input_reconstructed)
	list_of_values = [pearson_coeff_str, spearman_coeff_str, pearson_mean_genes_str, pearson_variance_genes_str, spearman_mean_genes_str, spearman_variance_genes_str, 
		average_variance_str, pearson_coeff_str_samples, spearman_coeff_str_samples, pearson_mean_samples_str, pearson_variance_samples_str, 
		spearman_mean_samples_str, spearman_variance_samples_str]
	#list_of_values = [pearson_coeff_str, spearman_coeff_str]
	print(list_of_values)
	#save list of all values
	output_file_name = output_directory_name + '/summaryOfAllCorrelations.txt'

	file = open(output_file_name, 'w')

	for value in list_of_values:
		file.write("%s\n" % value)


#get_summary(x_test, reconstructed_input, output_directory_name)


#--------------------------- Correlations 2d --------------------------------
#A is the actual input, B is the reconstructed input
def corr_2d(A,B):	#A is a datafrmae, B is a dataframe
	A_mA = A - A.mean(1)[:,None]
	B_mB = B - B.mean(1)[:,None]
	ssA = (A_mA**2).sum(1);
	ssB = (B_mB**2).sum(1);
	temp_corr = np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
	mean_corr = np.diagonal(temp_corr).mean()
	
	return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
