import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel, polynomial_kernel
import os
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
from source import Feature_Evaluation_multitask_SVM


def main():
    

if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)
    print('Start:', ncv.generate_timestamp())
    main()



