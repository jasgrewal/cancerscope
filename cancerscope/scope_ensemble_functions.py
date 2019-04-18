import copy
import pandas as pd
import numpy as np
import lasagne
### List of models
def getmodelsdict():
        modelOptions = {}
        with open(SCOPEMODELS_LIST, 'r') as f:
                for line in f:
                        if line.strip()!='':
                                modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')

                                modelOptions[modelname] = (url, expectedFile, expectedmd5)
        return modelOptions


### Function that enables removal of specific labels, or to sort the prediction dict by numeric (label prediction) or alphabet (labels)
def drop_sort_class(matrix4d_of_preds, label_to_drop=None, sortorder="alphabets"):
        ### This function takes in a matrix form of the dictionary containing individual models (shape num_models, num_samples, num_features, feature_tuples)
        ### It will return either a matrix form of the dictionary with the specified labels to drop removed, 
        ### Or optionally can return a amtrix form of the dictionary where the tuples are ordered by either 'numbers' or 'alphabets'
        if sortorder == 'numbers':
                ix_in_tuple_for_order = 1
        else:
                ix_in_tuple_for_order = 0
        df_matrix = copy.deepcopy(matrix4d_of_preds)
        if label_to_drop is None:
                for i in range(0, df_matrix.shape[0]):
                        for j in range(0, df_matrix.shape[1]):
                                sample_list_of_lab_tuples = df_matrix[i][j]
                                new_alphabet_order = sorted(sample_list_of_lab_tuples.tolist(), key=lambda x:-float(x[ix_in_tuple_for_order]))
                                df_matrix[i][j] = new_alphabet_order
                return df_matrix
        else:
                if ix_in_tuple_for_order == 1:
                        # We need to order the matrix alphabetically prior to deleting any labels
                        df_matrix = drop_sort_class(matrix4d_of_preds, label_to_drop=None, sortorder='alphabets')
                for p in label_to_drop:
                        ## DO something to drop the column
                        dropsite_labelix = np.where(df_matrix[0,0,:,0]==p)[0].tolist()
                        df_matrix = np.delete(df_matrix, dropsite_labelix, axis=2)
                return drop_sort_class(df_matrix, label_to_drop=None, sortorder=sortorder)

### Function that takes in dict and returns the top voted classes per sample and corresponding predictions
def get_ensemble_score(dict_with_preds, ignorelabs_list=None, x_sample_names=None):
        ### This function assumes the label,value tuples for each sample, for each model, are ordered by highest value (ix 0) to lowest value (ix -1) 
        modelnames = dict_with_preds.keys()
        df_temp_matrix = np.array([dict_with_preds[i] for i in modelnames])
        df_matrix = drop_sort_class(df_temp_matrix, label_to_drop=ignorelabs_list, sortorder='numbers')
        ### First get a list of predicted labels and values
        num_models, num_samples, num_preds, num_tuples_preds = df_matrix.shape
        flat_top_preds_values = df_matrix[:, :,0, 1].transpose().flatten() # Across all models, across all samples, the 0th ordered (top level) prediction, numeric value
        flat_top_preds_labels = df_matrix[:, :,0, 0].transpose().flatten() # Across all models, across all samples, the 0th ordered (top level) prediction, label
        ## Combine these into a sensible dataframe so as to separate each sample
        topPreds_bymodels_df = pd.DataFrame(np.column_stack([flat_top_preds_labels, flat_top_preds_values, modelnames * num_samples]), columns=['label', 'pred', 'modelname'])
        topPreds_bymodels_df['sample_ix'] = [np.mod(m/num_models, num_models) for m in topPreds_bymodels_df.index.tolist()]
        topPreds_bymodels_df[['pred']] = topPreds_bymodels_df[['pred']].astype(float) # dtype conversion for confidence scores
        ## Aggregate based on the predicted labels
        avg_per_label = topPreds_bymodels_df.groupby(['sample_ix', 'label'], as_index=False).mean()
        modelnames_label = topPreds_bymodels_df.groupby(['sample_ix', 'label'], as_index=False)['modelname'].apply(lambda x: "%s" % ','.join(x)).reset_index().rename(columns={0:"models"})
        modelnames_count = topPreds_bymodels_df.groupby(['sample_ix', 'label'], as_index=False)['modelname'].count().rename(columns={"modelname":"freq"})
        joined_df_list = [avg_per_label, modelnames_count ,modelnames_label]
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['sample_ix','label'], how='outer'), joined_df_list)
        df_merged.sort_values(by=['sample_ix', 'freq', 'pred'], ascending=False).sort_values('sample_ix').reset_index(drop=True)
        df_merged["rank_pred"] = df_merged.index
        for s in range(0, num_samples):
                df_merged.loc[df_merged.sample_ix == s, "rank_pred"] =  range(1, df_merged[df_merged.sample_ix == s].shape[0]  +1)
        if x_sample_names is not None:
                df_merged['sample_name'] = [x_sample_names[m] for m in df_merged['sample_ix'].tolist()]
        return df_merged

### Function to set up a custom network
def build_custom_mlp(n_out, num_features, depth, width, drop_input, drop_hidden, input_var=None, is_image=False):
        if is_image:
                network = lasagne.layers.InputLayer(shape=(None, 1, num_features,1), input_var=input_var)
        else:
                network = lasagne.layers.InputLayer(shape=(None,num_features), input_var=input_var)
        if drop_input:
                network = lasagne.layers.dropout(network, p=drop_input)
        nonlin = lasagne.nonlinearities.tanh
        for _ in range(depth):
                network=lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
                if drop_hidden:
                        network = lasagne.layers.dropout(network, p=drop_hidden)
        #Final output layer is softmax, for multiclass output
        softmax = lasagne.nonlinearities.softmax
        network = lasagne.layers.DenseLayer(network, n_out, nonlinearity=softmax)
        return network


