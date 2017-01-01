import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import time

class SVD_test:
    def __init__(self, col, row, num):
        self.col = tf.constant(col)
        self.row = tf.constant(row)
        self.num = tf.constant(num)
        self.m = np.random.random((row, col))
        self.matrix = tf.Variable(self.m)
        self.init_op = tf.global_variables_initializer()
		
    def single_process_svd(self, matrix):
        s,u,v = tf.svd(matrix, full_matrices=False)
        m1 = tf.matmul(u,tf.diag(s))
        m_est = tf.matmul(m1,v,transpose_b=True)

        with tf.Session() as sess:
            sess.run(self.init_op)
            start_time = time.time()
            result = sess.run([u,s,v])
            elapsed_time = time.time()-start_time
            print("elapsed_time",elapsed_time)
            estimation = sess.run(m_est)
            return estimation
			
if __name__ == "__main__":
    SVD = SVD_test(1000,1000,1)
    resultSingle = SVD.single_process_svd(SVD.m)
    print("error: ",LA.norm(resultSingle-SVD.m))
