import sys 
import os
sys.path.append("../../")

from memEfficientEvolve2DLattice import getMeasurementMeanVarSkew

# dir = '/mnt/talapasShared'

# for alpha in os.listdir(dir):
#     alpha_dir = os.path.join(dir, alpha, 'L5000', 'tMax10000')
#     alpha_val = alpha.replace("alpha", "")
#     print(alpha_val)
#     if not os.path.exists(alpha_dir):
#         continue
#     getMeasurementMeanVarSkew(alpha_dir, alpha_val)

dir = '/mnt/talapas/2DSqrtLogT/symmetricDirichlet/1,1,1,1'

getMeasurementMeanVarSkew(dir, 'symmetric')