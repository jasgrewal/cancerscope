import pkg_resources

def main():
	"""Test correct versions installed for theano and lasagne"""
	assert pkg_resources.get_distribution("theano").version == '1.0.4+unknown'
	assert pkg_resources.get_distribution("lasagne").version == "0.2.dev1"

if __name__ == "__main__":
	main()

