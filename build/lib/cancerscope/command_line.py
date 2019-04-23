import cancerscope
import os, sys
import tempfile

""" Need to bind raw_input to input in Python 2.* """
try:
	input = raw_input
except NameError:
	pass

def main():
	tempDir = "data/" #tempfile.mkdtemp()
	try:
		targetdir = input("Cancerscope requires pretrained models to be downloaded from the web. By default, these models will be downloaded to {0}.\nAlternatively, please enter a directory of your choice. A subdirectory named 'cancerscope_models' will be created there.\t".format(tempDir))
	except (SyntaxError):
		targetdir = tempDir
	
	""" raw_input() in python 2.* returns an empty string if user presses Return without input """
	if targetdir == "":
		targetdir = tempDir
	
	""" Now we can proceed with checking if the directory exists, and if not, raise error """
	if os.path.exists(targetdir):
		targetdir = targetdir + "/cancerscope_models/"
		if not os.path.exists(targetdir):
			os.mkdir(targetdir)
	else:
		print(targetdir is None)
		print("Please re-enter the target directory as {0} does not exist".format(targetdir))
		main(); return()
	
	print("Downloading data files to {0}".format(targetdir))
	cancerscope.get_models.getmodel(targetdir=targetdir)

if __name__ == "__main__":
	main()
	
