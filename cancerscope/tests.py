def test_setup(log=print):
	log("We need to check for all dependencies first")
	
def get_model_path(modelname):
	print("Need to map model name to model path, missing file")

def get_model(modelnpz, log=print):
	with np.load(modelnpz) as f:
		params = [f['arr_%d' % i] for i in range(len(f.files))]
	return params

def __predict__():
	return('Currently under development')


