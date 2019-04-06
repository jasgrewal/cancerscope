import os
import argparse

#Define basic params used during execution
def roundup(x):
	return int(math.ceil(x / 1000.0)) * 1000

def restricted_float(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
                raise argparse.ArgumentTypeError("%r not i nrange [0.0, 1.0]"%(x,))
        else:
                return x

def proper_delim(x):
	delims_list = ('\t', ' ', ',')
	if not x in delims_list:
		raise argparse.ArgumentTypeError("%r is not a valid Delimiter" % (x, ))
		raise argparse.ArgumentTypeError('Possible values {0}'.format(delims_list))
	else:
		return x.upper()

def proper_genecol(x):
	genecol_list = ('HUGO', 'GAF', 'GENCODE', 'ENSG', 'HUGO_ENSG')
	if not x.upper() in genecol_list:
		raise argparse.ArgumentTypeError("%r is not a valid Gene Name type " % (x, ))
		raise argparse.ArgumentTypeError('Possible values {0}'.format(genecol_list))
	else:
		return x.upper()

def proper_file(x):
	x=os.path.abspath(x)
        if not os.path.exists(x):
                raise argparse.ArgumentTypeError("%r is not a valid file"%(x,))
        else:
                return x

def proper_dir(x):
        if not os.path.exists(x):
                if not os.path.exists(os.path.join(x,"/../")):
                        raise argparse.ArgumentTypeError("%r is not a valid dir"%(x,))
                else:
                        os.makedirs(x)
                        return x
        else:
                return x

def Dictionary(labelset):
    with open(labelset) as f: d = dict((k.rstrip(), v.rstrip()) for k,v in (line.split('\t') for line in f))
    return (d)

def makeintifpossible(x):
    try:
        y = int(float(x))
    except:
        y = float('nan')

    return y

def int_or_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        x = int(x)
        if x > 1:
            return x
        raise argparse.ArgumentTypeError('$r is not a float in range [0,1], nor an int > 1' % (x,))
    else:
        return x

def roundup(x):
	return int(match.ceil(x / 1000.0)) * 1000

def check_norm_values(x):
    normlist = ('minmax', 'scale', 'normscale','none', 'rasterize','rastminmax')
    if x not in normlist:
        raise argparse.ArgumentTypeError('%r is not a valid normalization method. ' % (x,))
	raise argparse.ArgumentTypeError('Possible values {0}'.format(normlist))
    else:
        return x

def check_balance_type(x):
        if x not in ('none', 'weight', 'smote', 'dup','imblearn', 'adasyn'):
                raise argparse.ArgumentTypeError('%r is not a valid class expansion method. Possible values - none, weight, smote, dup , adasyn, imblearn' %(x,))
        else:
                return x

