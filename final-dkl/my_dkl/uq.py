import numpy as np

def ARC(trues, preds, confs):
	indices = np.argsort(confs)
	c = np.equal(trues, preds)[indices]
	rejs = np.arange(indices.shape[0]+1)/indices.shape[0]
	accs = []
	for i in range(indices.shape[0]):
		accs.append( np.mean(c[i:]) )
	accs = np.array(accs + [1])
	return rejs, accs

def ARC_L1_score(rejs, accs):
	A_oracle = accs[0] - accs[0]*np.log(accs[0])
	A_arc = (rejs[1:]-rejs[:-1]).dot(accs[:-1]+accs[1:])/2
	return A_oracle - A_arc
