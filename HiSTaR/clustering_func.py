import scanpy as sc
import pandas as pd
import numpy as np


def mclust_R(adata, n_clusters, use_rep='HiSTaR', key_added='HiSTaR', random_seed=2023):
    modelNames = 'EEE'
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


def configure_r_environment():
    import ctypes

    try:
        ctypes.CDLL(r"D:\R-4.4.2\bin\x64\Rblas.dll")
        ctypes.CDLL(r"D:\R-4.4.2\bin\x64\Rlapack.dll")
        # print("Rblas.dll and Rlapack.dll load successful！")
    except Exception as e:
        print(f"load error：{e}")

    import os
    from pathlib import Path
    import ctypes
    r_bin = Path("D:/R-4.4.2/bin/x64")
    ctypes.CDLL(str(r_bin / "Rblas.dll"))
    ctypes.CDLL(str(r_bin / "Rlapack.dll"))

    os.environ['R_HOME'] = str(r_bin.parent.parent)
    os.environ['PATH'] = f"{r_bin};{os.environ.get('PATH', '')}"

    import rpy2.robjects as robjects
    from rpy2.rinterface_lib import openrlib
    openrlib.rlib.R_set_command_line_arguments(0, [])

    test_code = '''
        x <- matrix(c(1,2,3,4), 2, 2)
        # print("Testing simple SVD:")
        try({
            result <- La.svd(x)
            # print(result)
        }, silent=FALSE)
    '''
    result = robjects.r(test_code)
    # print(result)

    os.environ['R_HOME'] = 'D:/R-4.4.2'
    os.environ['R_USER'] = os.path.expanduser('~')
    os.environ['PATH'] = 'D:/R-4.4.2/bin/x64;' + os.environ['PATH']
    os.environ['R_LIBS'] = 'D:/R-4.4.2/library'