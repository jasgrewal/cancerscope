from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='cancerscope', version='0.20', description='Pan-cancer classifier using RNA-Seq', url='http://github.com/jasgrewal/cancerscope/', author='Jasleen Grewal', author_email='grewalj23@gmail.com', license='MIT', packages=['cancerscope'], zip_safe=False, install_requires=['theano>=0.10.0b1', 'lasagne>=0.1'], classifiers=['Programming Language :: Python :: 2.7.14', 'Topic :: Machine Learning :: Genomics'], keywords='SCOPE Cancer Diagnostics Transcriptomics', include_package_data=True, long_description=readme(), test_suite='nose.collector', tests_require=['nose'], scripts=['bin/lasagne_SCOPE_testsample.py')


