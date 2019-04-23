"""
Basic utility functions for cancerscope
"""

import os
import zipfile, tarfile
import hashlib
import requests
import logging
import traceback
import shutil

def file_as_bytes(file):
	with file:
		return file.read()

def _calcMD5(filename):
	return hashlib.md5(file_as_bytes(open(filename, 'rb'))).hexdigest()


# From: https://stackoverflow.com/questions/32763720/timeout-a-file-download-with-python-urllib
def _downloadFile(url, filename, timeout=180):
	request = requests.get(url, timeout=10, stream=True)
	with open(filename, 'w') as fh:
		## Walk through request response in chunks of 1024*1024 bytes
		for chunk in request.iter_content(1024*10240):
			fh.write(chunk)

def _downloadFiles(files, downloadDir):
	if not os.path.exists(downloadDir):
		os.mkdir(downloadDir)
	
	for url, shortName, expectedMD5 in files:
		downloadedPath = os.path.join(downloadDir, os.path.basename(url)) # Where will the compressed file be downloaded
		targetPathDir = os.path.join(downloadDir, shortName) ## What is the destination folder name for extracted files
		if os.path.isfile(downloadedPath):
			downloadedMD5 = _calcMD5(downloadedPath) 
			if not downloadedMD5 == expectedDM5:
				os.remove(downloadedPath)
		
		if not os.path.isfile(downloadedPath):
			try:
				_downloadFile(url, downloadedPath)
			except Exception as e:
				logging.error(traceback.format_exc())
				print(type(e)); raise
			downloadedMD5 = _calcMD5(downloadedPath)
			assert downloadedMD5 == expectedMD5, "MD5 checksum mismatch with downloaded file: %s" % shortName
			
			if downloadedPath.endswith('tar.gz'):
				tar = tarfile.open(downloadedPath, 'r:gz')
				tar.extractall(path=downloadDir)
				tar.close()
			os.remove(downloadedPath)

