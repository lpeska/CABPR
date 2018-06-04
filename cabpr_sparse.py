"""
Content-adjusted Bayesian Personalized Ranking Matrix Factorization

Original Implementation of BPR obtained from https://github.com/gamboviol/bpr
Some IO operations and CV brought from PyDTI component, https://github.com/stephenliu0423/PyDTI

Extensions of BPR-OPT with arbitrary many Content Alignments by Ladislav Peska, peska@ksi.mff.cuni.cz
"""
from __future__ import division
import tensorflow as tf
import math
import itertools

from functions import *

class CABPR(object):

    def __init__(self,args):
        self.alg_type = args["alg_type"]
        self.batchSize = args["batch_size"]
        self.D = args["D"]
        self.orig_learning_rate = args["learning_rate"]
        self.learning_rate = self.orig_learning_rate
        self.max_iters = args["max_iters"]
        self.global_regularization = 0.05
        self.ca_regularization = 0.05
        self.ca_lambda = 0.05
        self.shape = args["shape"]
        self.fraction = args["fraction"]     
        self.learn_sim_weights = True 
        
        self.simple_predict = True
        
        self.sim_matrix_names = args["sim_names"]
        self.neg_item_learning_rate = 0.1
        self.k_size = 10
        self.sim_mat = {}
        self.np_sim_mat = {}
        self.np_sim_mat_csr = {}
        self.sim_user = {}
        self.sim_lambda = {}
        j=0
        for i in self.sim_matrix_names:
            self.np_sim_mat[i] = load_matrix(i, "data",self.shape,args["user_indicator"][j])
            self.sim_user[i] = args["user_indicator"][j]
            self.sim_lambda[i] = args["init_lambda"][j]
            j = j+1
        #regularize matrix similarities, so the average sum of similarity vector is 1
        for i in self.sim_matrix_names:
            self.np_sim_mat[i] = np.asmatrix(self.get_nearest_neighbors(self.np_sim_mat[i], self.k_size, i) )
            np.fill_diagonal(self.np_sim_mat[i], 0)
            uSum = (self.np_sim_mat[i].sum() / self.np_sim_mat[i].shape[0])
            self.np_sim_mat[i] = sp.coo_matrix(self.np_sim_mat[i])

            self.np_sim_mat[i] = (1 / uSum) * self.np_sim_mat[i]
            self.np_sim_mat_csr[i] = self.np_sim_mat[i].tocsr()


        if(len(self.sim_lambda)>0):            
            self.sim_lambda_zero = 1/len(self.sim_lambda)
        else:
            self.sim_lambda_zero = 1

        if self.learn_sim_weights == True:
            if self.alg_type == "TFLWMNN":
                self.filename = "nbpr_mcaTFLWMNN_sq %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
            elif self.alg_type == "TFLWM":
                self.filename = "nbpr_mcaTFLWM_sq %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
            elif self.alg_type == "TFLNN":
                self.filename = "nbpr_mcaTFLNN_sq %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
            elif self.alg_type == "TFL":
                self.filename = "nbpr_mcaTFL_sq %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
        else:
            self.filename = "nbpr_mcaTFL_sqnsw %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
        self.filename = self.filename.replace(" ", "_")
        self.filename = self.filename.replace(".", "")
        self.filename = self.filename.replace("D", ".")    
  
            
    def get_nearest_neighbors(self, S, size=5, name=""):
        if(name == "sim_Authors" or name == "sim_Books"):
            size = 50 #do not remove equally relevant items
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(np.asarray(S[i, :]).reshape(-1))[::-1][:min(size, n)]
            ii = ii[0:size] 
            if ii[0]>0:
                X[i, ii] = S[i, ii]
            
        return X   
    
    def learn_hyperparameters(self):    
        self.best_hyperpar_metric = 0
        self.best_params = {
            "max_iters":15,
            "global_regularization": 0.05,
            "ca_lambda": 0.05,
            "ca_regularization": 0.05,
        }
        W, test_data, test_label = cross_validation(self.dt, 8845, 1, 0, 0.1)

        #print(test_label[0:5000].tolist())
        #print(W[0, :].tolist())
        #print(type(test_label[0]))

        for glR in [0.01, 0.05]:
            if (len(self.sim_lambda) > 0) & (self.learn_sim_weights == True):
                for caL in [0.001, 0.01]:
                    for caR in [0.01, 0.05, 0.2]:
                        self.global_regularization = glR
                        self.ca_lambda = caL
                        self.ca_regularization = caR
                        self.learning_rate = self.orig_learning_rate
                        self.fix_model(W, self.dt, seed)
                        self.train(seed, test_data, test_label, True)
            elif (len(self.sim_lambda) > 0):
                for caL in [0.001, 0.01, 0.05]:
                    self.global_regularization = glR
                    self.ca_lambda = caL
                    self.learning_rate = self.orig_learning_rate
                    for i in self.sim_matrix_names:
                        self.sim_lambda[i] = self.sim_lambda_zero

                    self.fix_model(W, self.dt, seed)
                    self.train(seed, test_data, test_label, True)
            else:
                self.global_regularization = glR
                self.learning_rate = self.orig_learning_rate
                self.fix_model(W, self.dt, seed)
                self.train(seed, test_data, test_label, True)
                        
        self.max_iters = self.best_params["max_iters"]
        self.global_regularization = self.best_params["global_regularization"]
        self.ca_lambda = self.best_params["ca_lambda"] 
        self.ca_regularization = self.best_params["ca_regularization"]
        for i in self.sim_matrix_names:
            self.sim_lambda[i] = self.sim_lambda_zero
        self.learning_rate = self.orig_learning_rate
        
        print("finished learning with best hyperparameters iters: %s glR: %.3f caL: %.3f caR: %.3f"% (self.max_iters, self.global_regularization, self.ca_lambda,self.ca_regularization))

        
        
    def evaluate_hyperpar_results(self, aupr_val, auc_val, ndcg_val, iteration):
        hyperpar_metric =  ndcg_val
        if hyperpar_metric > self.best_hyperpar_metric:
            self.best_hyperpar_metric = hyperpar_metric
            self.best_params["max_iters"] = iteration
            self.best_params["global_regularization"] = self.global_regularization
            self.best_params["ca_lambda"] = self.ca_lambda
            self.best_params["ca_regularization"] = self.ca_regularization
            
        
    def fix_model(self, W, intMat, seed):
        self.learning_rate = self.orig_learning_rate
        self.num_users, self.num_objects = intMat.shape
        
        dt = np.multiply(W,intMat)  
        self.dt = dt
        #print(self.dt.shape)
        data = sp.csr_matrix(dt)
        self.data = data
        x, y = np.where(dt > 0)
        self.train_users, self.train_objects = set(x.tolist()), set(y.tolist())
        self.dinx = np.array(list(self.train_users))
        self.tinx = np.array(list(self.train_objects))

        self.dtTensor = tf.convert_to_tensor(self.dt)

        np.random.seed(seed)
                                                

    def train(self, seed, test_data, test_label, hyperpar_search = False):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        userSim: matrix of user similarities
        itemSim: matrix of item similarities
        """

        #close previous computation session, reinitialize variables to its original values
        if hasattr(self, 'sess'):
            self.sess.close()
            print("close session")

        tf.reset_default_graph()

        self.sess = tf.InteractiveSession()

        if self.alg_type == "TFLWMNN":
            self.initWithMarginAndNonNegativeSimWeight(seed)
        elif self.alg_type == "TFLWM":
            self.initWithMargin(seed)
        elif self.alg_type == "TFLNN":
            self.initWithNonNegativeSimWeight(seed)
        elif self.alg_type == "TFL":
            self.init(seed)

        initVars = tf.global_variables_initializer()
        self.sess.run(initVars)

        #act_loss = self.loss()
        n_samples = self.data.nnz
        #print 'initial loss {0}'.format(act_loss)
        t = time.clock()

        errorInLearning = False
        previousSimLambdaError = False


        for it in range(self.max_iters):
            users, pos_items, neg_items = self._uniform_user_sampling( n_samples)


            k = 0
            u_list = []
            i_list = []
            j_list = []
            for u,i,j in zip(users, pos_items, neg_items):
                k += 1
                u_list.append(u)
                i_list.append(i)
                j_list.append(j)
                if k % self.batchSize == 0:
                    err = self.update_factors(u_list, i_list, j_list)

                    if err == "sim_lambda overflow" and previousSimLambdaError == False:
                        previousSimLambdaError = True
                    elif err == "sim_lambda overflow" and previousSimLambdaError == True:
                        errorInLearning = True
                        break

                    u_list = []
                    i_list = []
                    j_list = []

            for k in self.sim_matrix_names:
                print(self.sess.run(self.sim_lambda[k]))

            if hyperpar_search == True and errorInLearning == True:
                break

            if(hyperpar_search != True):
                print(self.sim_lambda)

            

            t = time.clock()

            if self.learn_sim_weights == True:
                if self.alg_type == "TFLWMNN":
                    f = "train_nbpr_mcaTFLWMNN_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                elif self.alg_type == "TFLWM":
                    f = "train_nbpr_mcaTFLWM_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                elif self.alg_type == "TFLNN":
                    f = "train_nbpr_mcaTFLNN_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                elif self.alg_type == "TFL":
                    f = "train_nbpr_mcaTFL_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
            else:
                f = "train_nbpr_mcaTFL_sqnsw %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
            
            f = f.replace(" ", "_")
            f = f.replace(".", "")
            f = f.replace("D", ".") 
            if(hyperpar_search == True):
                if (it % 5 == 0) & (it != 0):
                    results = self.evaluation(test_data, test_label, True)
                    try:


                        self.evaluate_hyperpar_results(results[0], results[1], results[2], it)
                        with open(f, "a") as procFile:
                            procFile.write("iter:%.6f, glR: %.3f, caL: %.3f, caR: %.3f, aupr: %.5f,auc: %.5f,ndcg: %.5f\n" % (it,self.global_regularization, self.ca_lambda, self.ca_regularization,results[0], results[1], results[2]))
                    except Exception as e:
                        print(str(e))
                        """Do not evaluate results, write error"""
                        with open(f, "a") as procFile:
                            procFile.write(
                                "iter:%.6f, glR: %.3f, caL: %.3f, caR: %.3f, aupr: %s,auc: %s,ndcg: %s\n" % (
                                it, self.global_regularization, self.ca_lambda, self.ca_regularization, "NaN",
                                "NaN", "NaN"))

        if (hyperpar_search != True):
            if self.learn_sim_weights == True:
                if self.alg_type == "TFLWMNN":
                    fSW = "simWeights_nbpr_mcaTFLWMNN_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                elif self.alg_type == "TFLWM":
                    fSW = "simWeights_nbpr_mcaTFLWM_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                elif self.alg_type == "TFLNN":
                    fSW = "simWeights_nbpr_mcaTFLNN_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                elif self.alg_type == "TFL":
                    fSW = "simWeights_nbpr_mcaTFL_sq %.3f %s %s %s Dtxt" % (
                        self.fraction, self.simple_predict, self.k_size, len(self.sim_lambda))
                fSW = fSW.replace(" ", "_")
                fSW = fSW.replace(".", "")
                fSW = fSW.replace("D", ".")
                with open(fSW, "a") as swFile:
                    txt = ""
                    for k in self.sim_lambda:
                        txt = txt + "; " + k + ":" + str(self.sim_lambda[k].eval())
                    swFile.write("weights: %s\n" % txt )


    def sparse_slice(self, indices, values, needed_row_ids):
        needed_row_ids = tf.reshape(needed_row_ids, [1, -1])
        num_rows = tf.shape(indices)[0]
        resh = tf.cast(tf.reshape(indices[:, 0], [-1, 1]), tf.int32)
        partitions = tf.cast(tf.reduce_any(tf.equal(resh, needed_row_ids), 1), tf.int32)
        rows_to_gather = tf.dynamic_partition(tf.range(num_rows), partitions, 2)[1]
        slice_indices = tf.gather(indices, rows_to_gather)
        slice_values = tf.gather(values, rows_to_gather)
        return slice_indices, slice_values

    def initBPR(self, seed):
        self.num_users,self.num_items = self.data.shape
        self.item_bias = np.zeros(self.num_items)

        for i in self.sim_matrix_names:
            indices = np.mat([self.np_sim_mat[i].row, self.np_sim_mat[i].col]).transpose()
            self.sim_mat[i] = tf.SparseTensor(indices, self.np_sim_mat[i].data, self.np_sim_mat[i].shape)
            print(self.sim_mat[i])

        for i in self.sim_matrix_names:
            self.sim_lambda[i] = tf.get_variable("sim_lambda"+i, [1],dtype=tf.float64,
                   initializer=tf.constant_initializer(self.sim_lambda_zero))
            print("var "+i)

        if seed is None:
            self.user_factors = np.sqrt(1/float(self.D*self.num_users)) * np.random.normal(size=(self.num_users,self.D))
            self.item_factors = np.sqrt(1/float(self.D*self.num_items)) * np.random.normal(size=(self.num_items,self.D))
        else:
            prng = np.random.RandomState(seed)
            self.user_factors = np.sqrt(1/float(self.D*self.num_users)) * prng.normal(size=(self.num_users,self.D))
            self.item_factors = np.sqrt(1/float(self.D*self.num_items)) * prng.normal(size=(self.num_items,self.D))

        userf = tf.convert_to_tensor(self.user_factors)
        self.user_factors = tf.get_variable('user_factors', initializer=userf)

        itemf = tf.convert_to_tensor(self.item_factors)
        self.item_factors = tf.get_variable('item_factors', initializer=itemf)

        self.item_bias = tf.get_variable("item_bias", [self.num_items, 1],dtype=tf.float64,
                                 initializer=tf.constant_initializer(0.0))


        self.u = tf.placeholder(tf.int32, [self.batchSize])
        self.i = tf.placeholder(tf.int32, [self.batchSize])
        self.j = tf.placeholder(tf.int32, [self.batchSize])

        u_emb = tf.nn.embedding_lookup(self.user_factors, self.u)
        i_emb = tf.nn.embedding_lookup(self.item_factors, self.i)
        j_emb = tf.nn.embedding_lookup(self.item_factors, self.j)

        i_b = tf.nn.embedding_lookup(self.item_bias, self.i)
        j_b = tf.nn.embedding_lookup(self.item_bias, self.j)

        prediction_accuracy = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)

        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)),
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(j_emb, j_emb))
        ])

        bprloss = self.global_regularization * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(prediction_accuracy)))
        return bprloss,u_emb,i_emb,j_emb

    def initWithMarginAndNonNegativeSimWeight(self, seed):
        bprloss,u_emb,i_emb,j_emb = self.initBPR(seed)

        self.omega_fraction = round((self.num_users + self.num_objects) / 2)
        omega_fraction = tf.constant(self.omega_fraction, dtype=tf.float64)
        content_weight_regularization = tf.constant(self.ca_regularization, dtype=tf.float64)

        sim_lambda_zero = tf.constant(self.sim_lambda_zero, dtype=tf.float64)
        epsilon = tf.constant(0.1, dtype=tf.float64)
        zero = tf.constant(0.0, dtype=tf.float64)
        self.one = tf.constant(1.0, dtype=tf.float64)
        self.delta = tf.constant(1e-10, dtype=tf.float64)

        for k in self.sim_matrix_names:
            # update simmatrices
            if self.sim_user[k] == True:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx, sliced_vals = self.sparse_slice(indeces, values, self.u)
                uids, uidBars = tf.split(sliced_idx, 2, 1)
                uids = tf.squeeze(uids, [1])
                uidBars = tf.squeeze(uidBars, [1])

                uidFactors = tf.gather(self.user_factors, uids)
                uidBarFactors = tf.gather(self.user_factors, uidBars)
                user_difference = tf.subtract(uidFactors, uidBarFactors)
                user_difference = tf.maximum(
                    tf.reduce_sum(tf.multiply(user_difference, user_difference), 1, keep_dims=True) - epsilon, zero)
                weighted_user_difference = tf.multiply(user_difference, tf.transpose(sliced_vals))
                sim_user_error = tf.reduce_sum(weighted_user_difference)
                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * sim_user_error)

            else:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx_i, sliced_vals_i = self.sparse_slice(indeces, values, self.i)
                sliced_idx_j, sliced_vals_j = self.sparse_slice(indeces, values, self.j)


                oids_i, oidBars_i = tf.split(sliced_idx_i, 2, 1)
                oids_j, oidBars_j = tf.split(sliced_idx_j, 2, 1)
                oids_i = tf.squeeze(oids_i, [1])
                oids_j = tf.squeeze(oids_j, [1])
                oidBars_i = tf.squeeze(oidBars_i, [1])
                oidBars_j = tf.squeeze(oidBars_j, [1])


                oidFactors_i = tf.gather(self.item_factors, oids_i)
                oidBarFactors_i = tf.gather(self.item_factors, oidBars_i)
                oidFactors_j = tf.gather(self.item_factors, oids_j)
                oidBarFactors_j = tf.gather(self.item_factors, oidBars_j)


                item_difference_i = tf.subtract(oidFactors_i, oidBarFactors_i)
                item_difference_j = tf.subtract(oidFactors_j, oidBarFactors_j)


                item_difference_i = tf.maximum(tf.reduce_sum(tf.multiply(item_difference_i, item_difference_i), 1, keep_dims=True) - epsilon, zero)
                item_difference_j = tf.maximum(tf.reduce_sum(tf.multiply(item_difference_j, item_difference_j), 1, keep_dims=True) - epsilon, zero)

                weighted_item_difference_i = tf.multiply(item_difference_i, tf.transpose(sliced_vals_i))
                weighted_item_difference_j = tf.multiply(item_difference_j, tf.transpose(sliced_vals_j))


                sim_item_error_i = tf.reduce_sum(weighted_item_difference_i)
                sim_item_error_j = tf.reduce_sum(weighted_item_difference_j)


                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * (sim_item_error_i + (sim_item_error_j * self.neg_item_learning_rate)))

            sim_lambda_deviance = self.sim_lambda[k] - sim_lambda_zero
            lower_bound_error = sim_lambda_deviance * self.one / ( tf.maximum( self.sim_lambda[k], zero) + self.delta)
            upper_bound_error = sim_lambda_deviance * self.one / (tf.maximum( self.one - self.sim_lambda[k], zero) + self.delta)
            bprloss = bprloss + (content_weight_regularization * tf.multiply((lower_bound_error + upper_bound_error), (lower_bound_error + upper_bound_error) )) #omega_fraction *

        self.bprloss = bprloss
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.bprloss)


    def initWithNonNegativeSimWeight(self, seed):
        bprloss, u_emb, i_emb, j_emb = self.initBPR(seed)

        self.omega_fraction = round((self.num_users + self.num_objects) / 2)
        omega_fraction = tf.constant(self.omega_fraction, dtype=tf.float64)
        content_weight_regularization = tf.constant(self.ca_regularization, dtype=tf.float64)

        sim_lambda_zero = tf.constant(self.sim_lambda_zero, dtype=tf.float64)
        zero = tf.constant(0.0, dtype=tf.float64)
        self.one = tf.constant(1.0, dtype=tf.float64)
        self.delta = tf.constant(1e-10, dtype=tf.float64)

        for k in self.sim_matrix_names:
            # update simmatrices
            if self.sim_user[k] == True:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx, sliced_vals = self.sparse_slice(indeces, values, self.u)
                uids, uidBars = tf.split(sliced_idx, 2, 1)
                uids = tf.squeeze(uids, [1])
                uidBars = tf.squeeze(uidBars, [1])

                uidFactors = tf.gather(self.user_factors, uids)
                uidBarFactors = tf.gather(self.user_factors, uidBars)
                user_difference = tf.subtract(uidFactors, uidBarFactors)
                weighted_user_difference = tf.multiply(tf.reduce_sum(tf.multiply(user_difference, user_difference), 1), tf.transpose(sliced_vals))
                sim_user_error = tf.reduce_sum(weighted_user_difference)
                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * sim_user_error)

            else:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx_i, sliced_vals_i = self.sparse_slice(indeces, values, self.i)
                sliced_idx_j, sliced_vals_j = self.sparse_slice(indeces, values, self.j)


                oids_i, oidBars_i = tf.split(sliced_idx_i, 2, 1)
                oids_j, oidBars_j = tf.split(sliced_idx_j, 2, 1)
                oids_i = tf.squeeze(oids_i, [1])
                oids_j = tf.squeeze(oids_j, [1])
                oidBars_i = tf.squeeze(oidBars_i, [1])
                oidBars_j = tf.squeeze(oidBars_j, [1])


                oidFactors_i = tf.gather(self.item_factors, oids_i)
                oidBarFactors_i = tf.gather(self.item_factors, oidBars_i)
                oidFactors_j = tf.gather(self.item_factors, oids_j)
                oidBarFactors_j = tf.gather(self.item_factors, oidBars_j)

                item_difference_i = tf.subtract(oidFactors_i, oidBarFactors_i)
                item_difference_j = tf.subtract(oidFactors_j, oidBarFactors_j)


                weighted_item_difference_i = tf.multiply(tf.reduce_sum(tf.multiply(item_difference_i, item_difference_i),1), tf.transpose(sliced_vals_i))
                weighted_item_difference_j = tf.multiply(tf.reduce_sum(tf.multiply(item_difference_j, item_difference_j),1), tf.transpose(sliced_vals_j))


                sim_item_error_i = tf.reduce_sum(weighted_item_difference_i)
                sim_item_error_j = tf.reduce_sum(weighted_item_difference_j)


                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * (sim_item_error_i + (sim_item_error_j * self.neg_item_learning_rate)))
                

            sim_lambda_deviance = self.sim_lambda[k] - sim_lambda_zero
            lower_bound_error = sim_lambda_deviance * self.one / (tf.maximum(self.sim_lambda[k], zero) + self.delta)
            upper_bound_error = sim_lambda_deviance * self.one / (tf.maximum(self.one - self.sim_lambda[k], zero) + self.delta)
            bprloss = bprloss + (content_weight_regularization * tf.multiply((lower_bound_error + upper_bound_error), (
            lower_bound_error + upper_bound_error)))  # omega_fraction *

        self.bprloss = bprloss
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.bprloss)

    def initWithMargin(self, seed):
        bprloss,u_emb,i_emb,j_emb = self.initBPR(seed)

        self.omega_fraction = round((self.num_users + self.num_objects) / 2)
        omega_fraction = tf.constant(self.omega_fraction, dtype=tf.float64)
        content_weight_regularization = tf.constant(self.ca_regularization, dtype=tf.float64)

        sim_lambda_zero = tf.constant(self.sim_lambda_zero, dtype=tf.float64)
        epsilon = tf.constant(0.1, dtype=tf.float64)
        zero = tf.constant(0.0, dtype=tf.float64)

        for k in self.sim_matrix_names:
            # update simmatrices
            if self.sim_user[k] == True:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx, sliced_vals = self.sparse_slice(indeces, values, self.u)
                uids, uidBars = tf.split(sliced_idx, 2, 1)
                uids = tf.squeeze(uids, [1])
                uidBars = tf.squeeze(uidBars, [1])

                uidFactors = tf.gather(self.user_factors, uids)
                uidBarFactors = tf.gather(self.user_factors, uidBars)
                user_difference = tf.subtract(uidFactors, uidBarFactors)
                user_difference = tf.maximum(
                    tf.reduce_sum(tf.multiply(user_difference, user_difference), 1, keep_dims=True) - epsilon, zero)
                weighted_user_difference = tf.multiply(user_difference, tf.transpose(sliced_vals))
                sim_user_error = tf.reduce_sum(weighted_user_difference)
                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * sim_user_error)

            else:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx_i, sliced_vals_i = self.sparse_slice(indeces, values, self.i)
                sliced_idx_j, sliced_vals_j = self.sparse_slice(indeces, values, self.j)
                # print("sliced_idx_i")
                # print(sliced_idx_i,sliced_vals_i)

                oids_i, oidBars_i = tf.split(sliced_idx_i, 2, 1)
                oids_j, oidBars_j = tf.split(sliced_idx_j, 2, 1)
                oids_i = tf.squeeze(oids_i, [1])
                oids_j = tf.squeeze(oids_j, [1])
                oidBars_i = tf.squeeze(oidBars_i, [1])
                oidBars_j = tf.squeeze(oidBars_j, [1])

                # print("oids_i")
                # print(oids_i,oidBars_i)

                oidFactors_i = tf.gather(self.item_factors, oids_i)
                oidBarFactors_i = tf.gather(self.item_factors, oidBars_i)
                oidFactors_j = tf.gather(self.item_factors, oids_j)
                oidBarFactors_j = tf.gather(self.item_factors, oidBars_j)
                # print("oidFactors_i")
                # print(oidFactors_i, oidBarFactors_i)

                item_difference_i = tf.subtract(oidFactors_i, oidBarFactors_i)
                item_difference_j = tf.subtract(oidFactors_j, oidBarFactors_j)
                # print("item_difference_i")
                # print(item_difference_i, item_difference_j)

                item_difference_i = tf.maximum(tf.reduce_sum(tf.multiply(item_difference_i, item_difference_i), 1, keep_dims=True) - epsilon, zero)
                item_difference_j = tf.maximum(tf.reduce_sum(tf.multiply(item_difference_j, item_difference_j), 1, keep_dims=True) - epsilon, zero)

                weighted_item_difference_i = tf.multiply(item_difference_i, tf.transpose(sliced_vals_i))
                weighted_item_difference_j = tf.multiply(item_difference_j, tf.transpose(sliced_vals_j))


                sim_item_error_i = tf.reduce_sum(weighted_item_difference_i)
                sim_item_error_j = tf.reduce_sum(weighted_item_difference_j)
                # print(sim_item_error_j)

                # sim_item_error_i = tf.reduce_sum(tf.multiply(weighted_item_difference_i, weighted_item_difference_i))
                # sim_item_error_j = tf.reduce_sum(tf.multiply(weighted_item_difference_j, weighted_item_difference_j))

                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * (sim_item_error_i + (sim_item_error_j * self.neg_item_learning_rate)))


            sim_lambda_deviance = self.sim_lambda[k] - sim_lambda_zero
            bprloss = bprloss + (omega_fraction * content_weight_regularization * tf.multiply(sim_lambda_deviance,sim_lambda_deviance))

        self.bprloss = bprloss
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.bprloss)


    def init(self, seed):
        bprloss,u_emb,i_emb,j_emb = self.initBPR(seed)

        self.omega_fraction = round((self.num_users + self.num_objects) / 2)
        omega_fraction = tf.constant(self.omega_fraction, dtype=tf.float64)
        content_weight_regularization = tf.constant(self.ca_regularization, dtype=tf.float64)
        sim_lambda_zero = tf.constant(self.sim_lambda_zero, dtype=tf.float64)
        zero = tf.constant(0.0, dtype=tf.float64)

        for k in self.sim_matrix_names:
            # update simmatrices
            if self.sim_user[k] == True:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx, sliced_vals = self.sparse_slice(indeces, values, self.u)
                uids, uidBars = tf.split(sliced_idx, 2, 1)
                uids = tf.squeeze(uids, [1])
                uidBars = tf.squeeze(uidBars, [1])

                uidFactors = tf.gather(self.user_factors, uids)
                uidBarFactors = tf.gather(self.user_factors, uidBars)
                user_difference = tf.subtract(uidFactors, uidBarFactors)
                weighted_user_difference = tf.multiply(tf.reduce_sum(tf.multiply(user_difference, user_difference), 1), tf.transpose(sliced_vals))
                sim_user_error = tf.reduce_sum(weighted_user_difference)
                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * sim_user_error)

            else:
                indeces = self.sim_mat[k].indices
                values = self.sim_mat[k].values
                sliced_idx_i, sliced_vals_i = self.sparse_slice(indeces, values, self.i)
                sliced_idx_j, sliced_vals_j = self.sparse_slice(indeces, values, self.j)

                oids_i, oidBars_i = tf.split(sliced_idx_i, 2, 1)
                oids_j, oidBars_j = tf.split(sliced_idx_j, 2, 1)
                oids_i = tf.squeeze(oids_i, [1])
                oids_j = tf.squeeze(oids_j, [1])
                oidBars_i = tf.squeeze(oidBars_i, [1])
                oidBars_j = tf.squeeze(oidBars_j, [1])

                oidFactors_i = tf.gather(self.item_factors, oids_i)
                oidBarFactors_i = tf.gather(self.item_factors, oidBars_i)
                oidFactors_j = tf.gather(self.item_factors, oids_j)
                oidBarFactors_j = tf.gather(self.item_factors, oidBars_j)

                item_difference_i = tf.subtract(oidFactors_i, oidBarFactors_i)
                item_difference_j = tf.subtract(oidFactors_j, oidBarFactors_j)

                weighted_item_difference_i = tf.multiply(
                    tf.reduce_sum(tf.multiply(item_difference_i, item_difference_i), 1), tf.transpose(sliced_vals_i))
                weighted_item_difference_j = tf.multiply(
                    tf.reduce_sum(tf.multiply(item_difference_j, item_difference_j), 1), tf.transpose(sliced_vals_j))

                sim_item_error_i = tf.reduce_sum(weighted_item_difference_i)
                sim_item_error_j = tf.reduce_sum(weighted_item_difference_j)

                bprloss = bprloss + (self.sim_lambda[k] * self.ca_lambda * (
                sim_item_error_i + (sim_item_error_j * self.neg_item_learning_rate)))

            sim_lambda_deviance = self.sim_lambda[k] - sim_lambda_zero
            bprloss = bprloss + (omega_fraction * content_weight_regularization * tf.multiply(sim_lambda_deviance,sim_lambda_deviance))

        self.bprloss = bprloss
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.bprloss)


    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """One run of ADAM optimization"""
        #print("session run")
        loss_v = self.sess.run(self.train_step , feed_dict={
            self.u: u,
            self.i: i,
            self.j: j})

        returnText = ""

        if self.alg_type == "TFL" or self.alg_type == "TFLWM":
            sum_lambda = 0
            for k in self.sim_matrix_names:
                sum_lambda += abs(self.sim_lambda[k].eval())
                #print(sum_lambda,self.sim_lambda)
            for k in self.sim_matrix_names:
                if math.isnan(sum_lambda):
                    print("sim_lambda overflow")
                    tf.assign(self.sim_lambda[k], [self.sim_lambda_zero], validate_shape=False).eval()
                    returnText = "sim_lambda overflow"
                else:
                    tf.assign(self.sim_lambda[k], self.sim_lambda[k].eval()/sum_lambda).eval()
        else:
            for k in self.sim_matrix_names:
                val = self.sim_lambda[k].eval()
                if math.isnan(val[0]):
                    print("sim_lambda overflow")
                    tf.assign(self.sim_lambda[k], [self.sim_lambda_zero], validate_shape=False).eval()
                    returnText = "sim_lambda overflow"
                if val[0] <= 0.0:
                    tf.assign(self.sim_lambda[k], [self.delta], validate_shape=False).eval()
                elif val[0] >= 1.0:
                    tf.assign(self.sim_lambda[k], [self.one - self.delta], validate_shape=False).eval()

        return returnText


    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, 
          and then sample a positive and a negative item for each 
          user sample.
        """
        #print("Generating %s random training samples\n" % str(n_samples))
        
        sgd_users = np.random.choice(list(self.train_users),size=n_samples)
        sgd_ni = np.random.choice(list(self.train_objects),size=(n_samples*2))#*2 
        i = 0
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            pos_item = np.random.choice(self.data[sgd_user].indices)
            
            neg_item = sgd_ni[i]
            while neg_item in self.data[sgd_user].indices:
                i = i+1
                neg_item = sgd_ni[i]
                
            sgd_pos_items.append(pos_item)
            sgd_neg_items.append(neg_item)
            i = i+1

        return sgd_users, sgd_pos_items, sgd_neg_items        


    def log2(self, x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator

    def sparse_slice_ui(self, indices, values, oids, needed_row_ids):
        needed_row_ids = tf.reshape(needed_row_ids, [1, -1])

        resh = tf.cast(tf.reshape(indices[:, 0], [-1, 1]), tf.int32)
        eq = tf.equal(resh, needed_row_ids)

        idx = tf.split(tf.where(eq),2,1)

        slice_uids = tf.gather(tf.reshape(needed_row_ids, [-1]), idx[1])
        slice_uidBars = tf.gather(indices[:, 1], idx[0])
        slice_values = tf.gather(values, idx[0])
        slice_oids = tf.gather(oids, idx[1])


        return slice_uids, slice_uidBars, slice_values, slice_oids

    def evaluation(self, test_data, test_label, hyperpar_search = False):

        with tf.device('/cpu:0'):
            test_u = tf.placeholder(tf.int32, [None])
            test_i = tf.placeholder(tf.int32, [None])

            test_u_emb = tf.nn.embedding_lookup(self.user_factors, test_u)
            test_i_emb = tf.nn.embedding_lookup(self.item_factors, test_i)
            test_bias_emb = tf.nn.embedding_lookup(self.item_bias, test_i)



            scoresOp = test_bias_emb + tf.reduce_sum(tf.multiply(test_u_emb, test_i_emb), 1, keep_dims=True)

            chunkNumber = 10

            x, y = test_data[:, 0], test_data[:, 1]

            test_data1 = np.array_split(test_data,chunkNumber)
            scores = []
            curr_sim_weights = {}

            if self.simple_predict == False:
                trainUsers = tf.convert_to_tensor(list(self.train_users))
                for k in self.sim_matrix_names:
                    if self.sim_user[k] == True:
                        curr_sim_weights[k] = self.sim_lambda[k].eval()
                print(curr_sim_weights)

            for td in test_data1:
                if self.simple_predict == True:
                    scr = self.sess.run(scoresOp, feed_dict={
                        test_u: td[:, 0],
                        test_i: td[:, 1]})
                    scores.extend(list(itertools.chain.from_iterable(scr.tolist())))
                else:
                    tu = tf.convert_to_tensor(td[:, 0])
                    ti = tf.convert_to_tensor(td[:, 1])

                    tu_exp = tf.expand_dims(tu, -1)
                    inTrainLabel = tf.cast(tf.reduce_any(tf.equal(tu_exp, trainUsers), 1), tf.int32)

                    tu_parts = tf.dynamic_partition(tu, inTrainLabel,2)
                    ti_parts = tf.dynamic_partition(ti, inTrainLabel, 2)

                    tuIn = tu_parts[1]
                    tiIn = ti_parts[1]

                    uEmb = tf.nn.embedding_lookup(self.user_factors, tuIn)
                    oEmb = tf.nn.embedding_lookup(self.item_factors, tiIn)
                    bEmb = tf.nn.embedding_lookup(self.item_bias, tiIn)

                    scOp = bEmb + tf.reduce_sum(tf.multiply(uEmb, oEmb), 1, keep_dims=True)

                    scrIn = scOp.eval()

                    resOutU = tuIn.eval()
                    resOutI = tiIn.eval()
                    resOutS = scrIn

                    scr = {}

                    for idx, val in enumerate(resOutS):
                        ref = str(resOutU[idx])+"_"+str(resOutI[idx])
                        scr[ ref ] = val

                    for idx in range(td.shape[0]):
                        ref = str(td[idx, 0]) + "_" + str(td[idx, 1])
                        if ref in scr: #direct
                            scores.append(scr[ref])
                        else: #approximated user factors
                            u = td[idx, 0]
                            o = td[idx, 1]

                            users = []
                            weight = []
                            for k in self.sim_matrix_names:
                                if self.sim_user[k] == True:
                                    sm = self.np_sim_mat_csr[k]
                                    _u = sm.getrow(u).nonzero()[1]

                                    _w = curr_sim_weights[k] * sm[u, _u]
                                    users.extend(_u.tolist())
                                    weight.extend(_w.tolist())

                            objects = [o] * len(users)
                            scr = self.sess.run(scoresOp, feed_dict={
                                test_u: users,
                                test_i: objects})
                            scr = np.squeeze(scr, 1)
                            s = np.dot(scr, weight)
                            scores.append(s)


            scores = np.array(scores).T

            test_data = np.column_stack((x,y))
            test_label = np.array(test_label).T


            vals = per_user_rankings(test_data, test_label, scores)


            if hyperpar_search == False:
                with open(self.filename, "a") as procFile:
                    procFile.writelines(["%s;%s;%s;%s \n" % (item[0], item[1], item[2], item[3])  for item in vals.T])

        return np.mean(vals[1,:]), np.mean(vals[2,:]), np.mean(vals[0,:]), np.mean(vals[3,:])
    
    

        
    
    def __str__(self):
        return "Model: BPR_MCA, factors:%s, learningRate:%s,  max_iters:%s, global_reg:%s, bias_reg:%s, ca_reg:%s, simple_predict:%s" % (self.D, self.learning_rate, self.max_iters,  self.global_regularization, self.bias_regularization, self.ca_regularization,  self.simple_predict)
    


if __name__ == "__main__":  

    import time
    import numpy as np
    import pandas as pd
    

    #run CABPR on ML1M; parameters of the best resulting method out of the evaluated ones

    args = {
        "alg_type": "TFLNN",
        "D": 20,
        "learning_rate": 0.1,
        "max_iters":31,
        "batch_size":32,

        "sim_names": ["sim_userML1M", "sim_userUSPost", "sim_itemML1M", "sim_itemIMDB", "sim_itemDBT"],
        "user_indicator": [True, True, False, False, False],
        "init_lambda": [0.2, 0.2, 0.2, 0.2, 0.2]

    }
    fractionList = [0.98,0.95,0.90]

    # run CABPR on LOD-RecSys; parameters of the best resulting method out of the evaluated ones
    """
    args = {
        "alg_type" : "TFLNN",
        "D": 20,
        "learning_rate": 0.1,
        "max_iters":31,
        "batch_size":32,

        "sim_names": ["sim_Authors",  "sim_Books",  "sim_BroadCats",  "sim_Cats",  "sim_GenSim"],
        "user_indicator": [False, False, False, False, False],
        "init_lambda": [0.2, 0.2, 0.2, 0.2, 0.2]
    }
    fractionList = [0.90,0.75,0.50]
    """

    #load ML1M interaction data
    dt = pd.read_csv("data/flat_ratings.csv", sep=';', header=None)
    dt.columns = ['rows', 'cols', 'vals']

    coo_mat = sp.coo_matrix((dt['vals'],(dt['rows'],dt['cols'])))
    sm = coo_mat.tocsr()
    sma = sm.toarray()

   
    seedlist = [35, 2085, 1737, 8854, 124]
    for fraction in fractionList:
        args["shape"] = sma.shape
        args["fraction"] = fraction
        for seed in seedlist:
                # perform 5x Monte Carlo cross-validation
                W, test_data, test_label = cross_validation(sma, seed, 1, 0, fraction)  #


                model = CABPR(args)
                print(model.alg_type)
                print(fraction)
                print(seed)

                try:
                        model.fix_model(W, sma, 50)
                        model.learn_hyperparameters()
                        model.train(50, test_data, test_label)
                        with tf.device('/cpu:0'):
                            aupr_val, auc_val, ndcg_val, p10_val = model.evaluation(test_data, test_label)

                        print(aupr_val, auc_val, ndcg_val, p10_val)
                except Exception as e:
                        print(str(e))
                        print("Error during evaluation of fraction:%s, seed:%s" %(str(fraction), str(seed)))




