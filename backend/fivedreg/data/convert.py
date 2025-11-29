import pickle, numpy as np, sys
X,y=pickle.load(open(sys.argv[1],"rb")); np.savez(sys.argv[2], X=X, y=y)

