import numpy as np
import time
from pyDOE import lhs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, INLET, OUTLET, WALL, AREA_nonCYLD, CYLD, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        self.mu = 0.0025

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]

        self.x_AREA_nonCYLD = AREA_nonCYLD[:, 0:1]
        self.y_AREA_nonCYLD = AREA_nonCYLD[:, 1:2]

        self.x_CYLD = CYLD[:, 0:1]
        self.y_CYLD = CYLD[:, 1:2]

        # Define layers
        self.uv_layers = uv_layers

        # Initialize loss recording
        self.loss_rec = []
        self.loss_f_rec = []
        self.loss_WALL_rec = []
        self.loss_INLET_rec = []
        self.loss_OUTLET_rec = []
        self.loss_f_Euler_rec = []
        self.loss_CYLD_rec = []

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_AREA_nonCYLD_tf = tf.placeholder(tf.float32, shape=[None, self.x_AREA_nonCYLD.shape[1]])
        self.y_AREA_nonCYLD_tf = tf.placeholder(tf.float32, shape=[None, self.y_AREA_nonCYLD.shape[1]])

        self.x_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.x_CYLD.shape[1]])
        self.y_CYLD_tf = tf.placeholder(tf.float32, shape=[None, self.y_CYLD.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, self.f_pred_Euler_x, self.f_pred_Euler_y = self.net_uv(self.x_tf, self.y_tf)
        self.e1, self.e2, self.e3 = self.net_f(self.x_c_tf, self.y_c_tf)
        self.u_WALL_pred, self.v_WALL_pred, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        _, _, self.p_OUTLET_pred, _, _  = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)
        _, _, _, self.f_pred_AREA_nonCYLD_Euler_x, self.f_pred_AREA_nonCYLD_Euler_y  = self.net_uv(self.x_AREA_nonCYLD_tf, self.y_AREA_nonCYLD_tf)
        self.u_pred_CYLD, self.v_pred_CYLD, _, _, _ = self.net_uv(self.x_CYLD_tf, self.y_CYLD_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.e1)) \
                      + tf.reduce_mean(tf.square(self.e2))\
                      + tf.reduce_mean(tf.square(self.e3))
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred)) \
                       + tf.reduce_mean(tf.square(self.v_WALL_pred))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_mean(tf.square(self.v_INLET_pred-self.v_INLET_tf))
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred))
        self.loss_f_Euler = tf.reduce_mean(tf.square(self.f_pred_AREA_nonCYLD_Euler_x)) \
                         + tf.reduce_mean(tf.square(self.f_pred_AREA_nonCYLD_Euler_y))
        self.loss_CYLD = tf.reduce_mean(tf.square(self.u_pred_CYLD)) \
                       + tf.reduce_mean(tf.square(self.v_pred_CYLD))

        self.loss = self.loss_f + self.loss_f_Euler+ 2*self.loss_CYLD+ 2*(self.loss_WALL + self.loss_INLET + self.loss_OUTLET)

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

    def save_NN(self, fileDir):

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float32)
                b = tf.Variable(uv_biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        u = psips[:, 0:1]
        v = psips[:, 1:2]
        p = psips[:, 2:3]
        f_Euler_x = psips[:, 3:4]
        f_Euler_y = psips[:, 4:5]
        return u, v, p, f_Euler_x, f_Euler_y

    def net_f(self, x, y):

        rho = self.rho
        mu = self.mu
        u, v, p, f_Euler_x, f_Euler_y = self.net_uv(x, y)

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        e1 = (u * u_x + v * u_y) + (1 / rho) * p_x - (mu / rho) * (u_xx + u_yy) - (1 / rho) * f_Euler_x
        e2 = (u * v_x + v * v_y) + (1 / rho) * p_y - (mu / rho) * (v_xx + v_yy) - (1 / rho) * f_Euler_y
        e3 = u_x + v_y

        return e1, e2, e3

    def callback(self, loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET, loss_f_Euler, loss_CYLD):
        self.count = self.count+1
        self.loss_rec.append(loss)
        self.loss_f_rec.append(loss_f)
        self.loss_WALL_rec.append(loss_WALL)
        self.loss_INLET_rec.append(loss_INLET)
        self.loss_OUTLET_rec.append(loss_OUTLET)
        self.loss_f_Euler_rec.append(loss_f_Euler)
        self.loss_CYLD_rec.append(loss_CYLD)

        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET,
                   self.x_AREA_nonCYLD_tf: self.x_AREA_nonCYLD, self.y_AREA_nonCYLD_tf: self.y_AREA_nonCYLD,
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD,
                   self.learning_rate: learning_rate}

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))

            self.loss_rec.append(self.sess.run(self.loss, tf_dict))
            self.loss_f_rec.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_WALL_rec.append(self.sess.run(self.loss_WALL, tf_dict))
            self.loss_INLET_rec.append(self.sess.run(self.loss_INLET, tf_dict))
            self.loss_OUTLET_rec.append(self.sess.run(self.loss_OUTLET, tf_dict))
            self.loss_f_Euler_rec.append(self.sess.run(self.loss_f_Euler, tf_dict))
            self.loss_CYLD_rec.append(self.sess.run(self.loss_CYLD, tf_dict))

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET,
                   self.x_AREA_nonCYLD_tf: self.x_AREA_nonCYLD, self.y_AREA_nonCYLD_tf: self.y_AREA_nonCYLD,
                   self.x_CYLD_tf: self.x_CYLD, self.y_CYLD_tf: self.y_CYLD}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_f, self.loss_WALL, self.loss_INLET, self.loss_OUTLET, self.loss_f_Euler, self.loss_CYLD],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star})
        f_Euler_star_x = self.sess.run(self.f_pred_Euler_x, {self.x_tf: x_star, self.y_tf: y_star})
        f_Euler_star_y = self.sess.run(self.f_pred_Euler_y, {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star, f_Euler_star_x, f_Euler_star_y

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    # delete points within cylinder
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]

def ColCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    # Collect points within cylinder
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst<=r,:]

def CartGrid(xmin, xmax, ymin, ymax, num_x, num_y):
    # num_x, num_y: number per edge
    x = np.linspace(xmin, xmax, num=num_x)
    y = np.linspace(ymin, ymax, num=num_y)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy

if __name__ == "__main__":

    # Domain bounds
    lb = np.array([0, 0])
    ub = np.array([1.1, 0.40])

    # Network configuration
    uv_layers = [2] + 6*[100] + [7]

    # WALL = [x, y], u=v=0
    wall_up = [0.0, 0.40] + [1.1, 0.0] * lhs(2, 881)
    wall_lw = [0.0, 0.00] + [1.1, 0.0] * lhs(2, 881)

    # INLET = [x, y, u, v]
    U_max = 1.0
    INLET = [0.0, 0.0] + [0.0, 0.40] * lhs(2, 401)
    y_INLET = INLET[:,1:2]
    u_INLET = 4*U_max*y_INLET*(0.40-y_INLET)/(0.40**2)
    v_INLET = 0*y_INLET
    INLET = np.concatenate((INLET, u_INLET, v_INLET), 1)

    # OUTLET = [x, y], p=0
    OUTLET = [1.1, 0.0] + [0.0, 0.40] * lhs(2, 401)

    WALL = np.concatenate((wall_up, wall_lw), 0)

    x_AREA, y_AREA = CartGrid(xmin=0.0025, xmax=1.0975,
                                ymin=0.0025, ymax=0.3975,
                                num_x=439, num_y=159)
    AREA = np.concatenate((x_AREA, y_AREA), 1)
    AREA_CYLD = ColCylPT(AREA, xc=0.2, yc=0.2, r=0.055)
    AREA_nonCYLD = DelCylPT(AREA, xc=0.2, yc=0.2, r=0.055)
    AREA_nonCYLD = np.concatenate((AREA_nonCYLD, WALL, OUTLET, INLET[:,0:2]), 0)

    XY = np.concatenate((AREA_nonCYLD, AREA_CYLD), 0)

    print(XY.shape)

    # area exiting f_euler
    r = 0.05
    theta = (np.linspace(0, 2*np.pi, num=251)).flatten()[:, None]
    x_CYLD = np.multiply(r, np.cos(theta))+0.2
    y_CYLD = np.multiply(r, np.sin(theta))+0.2
    CYLD = np.concatenate((x_CYLD, y_CYLD), 1)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY[:,0:1], XY[:,1:2], marker='o', alpha=0.1 ,color='blue')
    plt.scatter(WALL[:,0:1], WALL[:,1:2], marker='o', alpha=0.2 , color='green')
    plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='o', alpha=0.2, color='orange')
    plt.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='o', alpha=0.2, color='red')
    plt.show()

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Train from scratch
        model = PINN_laminar_flow(XY, INLET, OUTLET, WALL, AREA_nonCYLD, CYLD, uv_layers, lb, ub)

        # Load trained neural network
        # model = PINN_laminar_flow(XY, INLET, OUTLET, WALL, AREA_nonCYLD, CYLD, uv_layers, lb, ub, ExistModel = 1, uvDir = 'uvNN.pickle')

        start_time = time.time()
        model.train(iter=10000, learning_rate=5e-4)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save neural network
        model.save_NN('uvNN.pickle')

        # Save loss history
        scipy.io.savemat('./loss.mat', {'loss':model.loss_rec,
                                   'loss_f':model.loss_f_rec,
                                   'loss_WALL':model.loss_WALL_rec,
                                   'loss_INLET':model.loss_INLET_rec,
                                   'loss_OUTLET':model.loss_OUTLET_rec,
                                   'loss_f_Euler':model.loss_f_Euler_rec,
                                   'loss_CYLD':model.loss_CYLD_rec})
