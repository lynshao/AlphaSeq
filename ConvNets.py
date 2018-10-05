from itertools import permutations
import tensorflow as tf
import random
import numpy as np
import pdb

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf.Session(config=config)

# from keras.backend.tensorflow_backend import set_session
# set_session(tf.Session(config=config))


class DeepNN():
    def __init__(self, args, stepSize):
        # game params
        self.N = args.N
        self.K = args.K
        self.M = args.M
        self.Q = args.Q
        self.stepSize = stepSize
        self.numfilters1 = args.numfilters1
        self.numfilters2 = args.numfilters2
        self.batchSize = args.batchSize
        self.l2_const = args.l2_const

        self.numMoves = self.Q ** self.stepSize

        # bulid DNN
        self.graph = tf.Graph()
        self.build_DNN()

        # initialization
        self.sess = tf.Session(graph = self.graph)
        self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))
        # self.sess.run(tf.global_variables_initializer())

        # save and load Params
        self.saver = tf.train.Saver(self.graph.get_collection('variables'))
        self.cost_his = []

        # policy entropy & cross entropy
        self.PolicyEntropy = 0
        self.CrossEntropy = 0
        self.cntEntropy = 0

    def build_DNN(self):
        # Neural Net
        with self.graph.as_default():
            # input placeholders
            self.batchInput = tf.placeholder(tf.float32, shape = [None, self.M, self.N, self.K])
            self.dropRate = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")

            x_img = tf.reshape(self.batchInput, [-1, self.N, self.K, self.M])

            # Conv0
            conv0 = tf.layers.conv2d(x_img, self.numfilters1, kernel_size=[3,3], padding='same')
            conv0 = tf.layers.batch_normalization(conv0, axis=-1, training=self.isTraining)
            conv0 = tf.nn.relu(conv0) # batchSize * N * K * self.numfilters1

            # Conv1
            conv1 = tf.layers.conv2d(conv0, self.numfilters1, kernel_size=[3,3], padding='same')
            conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.isTraining)
            conv1 = tf.nn.relu(conv1) # batchSize * N * K * self.numfilters1

            # Conv2
            conv2 = tf.layers.conv2d(conv1, self.numfilters1, kernel_size=[3,3], padding='same')
            conv2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.isTraining)
            conv2 = tf.nn.relu(conv2) # batchSize * N * K * self.numfilters1

            # Conv3
            conv3 = tf.layers.conv2d(conv2, self.numfilters1, kernel_size=[3,3], padding='same')
            conv3 = tf.layers.batch_normalization(conv3, axis=-1, training=self.isTraining)
            conv3 = tf.nn.relu(conv3) # batchSize * N * K * self.numfilters1

            # Output PiVec
            x4 = tf.layers.conv2d(conv3, 2, kernel_size=[1,1], padding='same')
            x4 = tf.layers.batch_normalization(x4, axis=-1, training=self.isTraining)
            x4 = tf.nn.relu(x4) # batchSize * (N-2) * (K-2) * self.numfilters1

            x4_flat = tf.reshape(x4, [-1, 2 * (self.N) * (self.K)])


            x5 = tf.layers.dense(x4_flat, self.numfilters2)
            x5 = tf.layers.batch_normalization(x5, axis=1, training=self.isTraining)
            x5 = tf.nn.relu(x5)
            x5_drop = tf.layers.dropout(x5, rate=self.dropRate) # batchSize x 1024

            self.piVec = tf.nn.softmax(tf.layers.dense(x5_drop, self.numMoves))

            # Output zValue
            y4 = tf.layers.conv2d(conv3, 1, kernel_size=[1,1], padding='same')
            y4 = tf.layers.batch_normalization(y4, axis=-1, training=self.isTraining)
            y4 = tf.nn.relu(y4) # batchSize * (N-2) * (K-2) * self.numfilters1

            y4_flat = tf.reshape(y4, [-1, 1 * (self.N) * (self.K)])


            y5 = tf.layers.dense(y4_flat, int(self.numfilters2/2))
            y5 = tf.layers.batch_normalization(y5, axis=1, training=self.isTraining)
            y5 = tf.nn.relu(y5)
            y5_drop = tf.layers.dropout(y5, rate=self.dropRate) # batchSize x 1024

            self.zValue = tf.nn.tanh(tf.layers.dense(y5_drop, 1)) 

            # calculate loss
            self.targetPi = tf.placeholder(tf.float32, shape=[None, self.numMoves])
            self.targetZ = tf.placeholder(tf.float32, shape=[None, 1])
            self.lr = tf.placeholder(tf.float32)
            self.loss_pi = tf.reduce_mean(-tf.reduce_sum(self.targetPi * tf.log(self.piVec), 1))
            # self.loss_pi =  tf.losses.softmax_cross_entropy(self.targetPi, self.piVec)
            self.loss_v = tf.losses.mean_squared_error(self.targetZ, self.zValue)

            # L2 regulization
            self.allParams = self.graph.get_collection('variables')
            l2_params = 0
            for paramsList in self.allParams:
                l2_params += tf.nn.l2_loss(paramsList)
            l2_penalty = self.l2_const * (l2_params * 2)


            self.total_loss = self.loss_pi + self.loss_v + l2_penalty
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

    def residual_block(self, input_layer, output_channel):

        conv1 = tf.layers.batch_normalization(input_layer, axis=-1, training=self.isTraining)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.conv2d(conv1, output_channel, kernel_size=[3,3], padding='same')

        
        conv2 = tf.layers.batch_normalization(conv1, axis=-1, training=self.isTraining)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.conv2d(conv2, output_channel, kernel_size=[3,3], padding='same')

        output = conv2 + input_layer
        return output

    def refresh_entropy(self):
        self.PolicyEntropy = 0
        self.CrossEntropy = 0
        self.cntEntropy = 0

    def output_entropy(self):
        return self.PolicyEntropy, self.CrossEntropy, self.cntEntropy

    def evaluate_node(self, rawstate, selfplay):
        state = self.feature_extract(rawstate)
        piVec, zValue = self.sess.run([self.piVec, self.zValue],  
                                    feed_dict= {self.batchInput:state,
                                    self.dropRate: 0,
                                    self.isTraining: False,
                                    })
        return piVec, zValue
    
    def update_DNN(self, mini_batch, lr):
        # expand mini_batch
        state_batch = np.array([data[0] for data in mini_batch])
        piVec_batch = np.array([data[1] for data in mini_batch])
        reward_batch = np.array([data[2] for data in mini_batch])[:,np.newaxis]
        state_batch = self.feature_extract(state_batch)
        _, batchLoss = self.sess.run([self.train_step, self.total_loss], 
                                feed_dict= {self.batchInput: state_batch,
                                self.dropRate: 0.3,
                                self.isTraining: True,
                                self.lr: lr,
                                self.targetPi: piVec_batch,
                                self.targetZ: reward_batch,
                                })
        self.cost_his.append(batchLoss)


    def feature_extract(self,state_batch):
        feature1 = np.copy(state_batch)
        feature1[feature1 == -1] = 0
        feature2 = np.copy(state_batch)
        feature2[feature2 == 1] = 0
        feature2[feature2 == -1] = 1
        feature3 = np.zeros(state_batch.shape)
        feature3[state_batch == 0] = 1

        state = np.reshape(np.hstack((feature1,feature2,feature3)),(len(feature1),self.M, self.N, self.K))
        return state

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def saveParams(self, path):
        self.saver.save(self.sess, path)

    def loadParams(self, path):
        self.saver.restore(self.sess, path)

    def get_params(self):
        return self.graph.get_collection('variables'), self.sess.run(self.graph.get_collection('variables'))
