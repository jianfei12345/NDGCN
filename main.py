import numpy as np
import jianfei_src

Dataset = np.load('Dataset1.npy', allow_pickle=True).item()

output = jianfei_src.InterAB_Run(Dataset, jianfei_src.NDGNN)



