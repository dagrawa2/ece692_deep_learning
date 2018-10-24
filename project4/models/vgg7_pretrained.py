import time
import tensorflow as tf
import numpy as np

class vgg7:

    def __init__(self, lr=0.01, mbs=1, pred_mbs=None, seed=None):
        self.lr = lr
        self.mbs = mbs
        self.pred_mbs = pred_mbs
        self.is_training = True
        if seed is not None: tf.set_random_seed(seed)
        self.build_graph()

    def convlayers(self):
        self.parameters = []

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_bn = tf.layers.batch_normalization(out, training=self.is_training)
            out_drop = tf.layers.dropout(out_bn, training=self.is_training)
            self.conv1_1 = tf.nn.relu(out_drop, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_bn = tf.layers.batch_normalization(out, training=self.is_training)
            self.conv1_2 = tf.nn.relu(out_bn, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_bn = tf.layers.batch_normalization(out, training=self.is_training)
            out_drop = tf.layers.dropout(out_bn, training=self.is_training)
            self.conv2_1 = tf.nn.relu(out_drop, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_bn = tf.layers.batch_normalization(out, training=self.is_training)
            out_drop = tf.layers.dropout(out_bn, training=self.is_training)
            self.conv2_2 = tf.nn.relu(out_drop, name=scope)
            self.parameters += [kernel, biases]

        # conv2_3
        with tf.name_scope('conv2_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out_bn = tf.layers.batch_normalization(out, training=self.is_training)
            self.conv2_3 = tf.nn.relu(out_bn, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')


    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool2.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 512],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool2_flat = tf.reshape(self.pool2, [-1, shape])
            pool2_flat_drop = tf.layers.dropout(pool2_flat, training=self.is_training)
            fc1l = tf.nn.bias_add(tf.matmul(pool2_flat_drop, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([512, 10],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc1_bn = tf.layers.batch_normalization(self.fc1, training=self.is_training)
            fc1_drop = tf.layers.dropout(fc1_bn, training=self.is_training)
            self.fc2l = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            self.parameters += [fc2w, fc2b]

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 
10])
        self.convlayers()
        self.fc_layers()
        self.logits = self.fc2l
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

    def start_session(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def stop_session(self):
        self.sess.close()

    def train(self, X_train, Y_train, eval_set=None, epochs=1, early_stopping=None):
        X_test, Y_test = eval_set
        if self.pred_mbs is None: self.pred_mbs = X_test.shape[0]
        n_train = X_train.shape[0]
        n_batches = int(np.ceil(n_train/self.mbs))
        mb_progress = int(0.1*n_batches)
        indices = np.arange(X_train.shape[0])
        accs = []
        if early_stopping is None: early_stopping = epochs+1
        for epoch in range(epochs):
            print("Epoch ", epoch, " . . . ")
            time_0 = time.time()
            np.random.shuffle(indices)
            X = X_train[indices]
            Y = Y_train[indices]
            time_1 = time.time()
            for k, i in enumerate(range(0,n_train,self.mbs)):
#                if k%mb_progress == 0:
#                    print(k, "/", n_batches, " batches (", np.round(time.time()-time_1, 5), " s)")
#                    time_1 = time.time()
                X_batch, Y_batch = X[i:min(n_train,i+self.mbs)], Y[i:min(n_train,i+self.mbs)]
                self.sess.run([self.train_step], feed_dict={self.x: X_batch, self.y_: Y_batch})
            accs.append(self.accuracy(Y_test, self.predict_logits(X_test)))
            print("\t Test accuracy: ", np.round(accs[-1], 6), " (", np.round(time.time()-time_0, 5), " s)")
            if 1+epoch >= early_stopping and np.argmax(accs[-early_stopping:])==0: break
        return np.array(accs)

    def predict_logits(self, X):
        self.is_training = False
        n_test = X.shape[0]
        Y_batches = []
        for i in range(0,n_test,self.pred_mbs):
            X_batch = X[i:min(n_test,i+self.pred_mbs)]
            Y_batches.append( self.sess.run(self.logits, feed_dict={self.x: X_batch}) )
        self.is_training = True
        return np.concatenate(tuple(Y_batches), axis=0)

    def accuracy(self, Y_true, Y_pred):
        return np.mean(np.argmax(Y_true, axis=1)==np.argmax(Y_pred, axis=1))

    def load_weights(self, filename):
        weights = np.load(filename)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
#            print i, k, np.shape(weights[k])
            self.sess.run(self.parameters[i].assign(weights[k]))
