from streaming_gmm.streaming_variational_gmm import StreamingVariationalGMM, VariationalGMM
from streaming_gmm.generate_synthetic_data import GMMDataGenerator
import numpy as np
M = 1
# Number of components per class.
K = 2
# Dimension of data.
D = 4
#np.random.seed()

synthetic_gmm = GMMDataGenerator(k=K, d=D)
print('Mu:\n', synthetic_gmm.get_mu())
print('Cov:\n', synthetic_gmm.get_cov())

n_batchs = 100
X_batchs = []
C_batchs = []
for i in range(n_batchs):
    new_X, new_C = synthetic_gmm.generate(n=20)
    X_batchs.append(new_X)
    C_batchs.append(new_C)


result_list = []
streaming_vb_gmm = StreamingVariationalGMM(K, D, max_iter=50, alpha_0=.5)
for X, C in zip(X_batchs, C_batchs): 
    streaming_vb_gmm.update_with_new_data(X[0].reshape(1, -1))
    #vbGmm = VariationalGMM(K, D)
    #vbGmm.fit(X, max_iter=20)
    #batch_result = streaming_vb_gmm.get_checkpoint()
    #result_list.append(streaming_vb_gmm.get_checkpoint())
result_list = streaming_vb_gmm.checkpoints
# plot_gmm(X, vbGmm.m_k, np.linalg.inv(vbGmm.nu_k[:, np.newaxis, np.newaxis]*vbGmm.W_k))