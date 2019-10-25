from common import *

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()

	return Variable(x, volatile=volatile)
