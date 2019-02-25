import os, gzip, pickle
def load_custom_data(dataset):
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
                new_path = os.path.join(
                 os.path.split(__file__)[0],
                "..",
                "data",
                dataset
                )
                if os.path.isfile(new_path):
                        dataset = new_path
         
        with gzip.open(dataset, 'rb') as f:
                try:
                        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
                except:
                        train_set, valid_set, test_set = pickle.load(f)
        
        return [train_set, valid_set, test_set]


