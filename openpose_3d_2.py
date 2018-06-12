import tensorflow as tf
import logging
import numpy as np
import cv2
import os
import glob
from slim_pose.network_cmu import CmuNetwork
from slim_pose.estimator import PoseEstimator
from slim_pose.lineal_model import lineal_model

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def post_est(outputs, batchsize, s=True):

    heat = outputs[:, :, :, :19]
    vect = outputs[:, :, :, 19:]
    human_parts = np.zeros([batchsize, 120, 18, 3])
    humnas_num = np.zeros([batchsize])
    for batch in range(batchsize):
        humans = PoseEstimator.estimate(heat[batch], vect[batch])
        humnas_num[batch] = len(humans)
        for h_num in range(len(humans)):
            for part_num in humans[h_num].body_parts.keys():
                x = humans[h_num].body_parts[part_num].x
                y = humans[h_num].body_parts[part_num].y
                score = humans[h_num].body_parts[part_num].score
                human_parts[batch, h_num, part_num, :] = np.array([y, x, score])
    if s:
        human_parts = human_parts[:, 0, :14, :2]
        human_parts = human_parts[:, :, ::-1].reshape(-1, 14 * 2)
    return human_parts.astype(np.float32), humnas_num.astype(np.int32)


def main():

    pretrain_path = '/media/rodrigo/c1d7e9c9-c8cb-402e-b241-9090925389b3/CMU_open_pose/tf-pose-estimation-master/models/numpy/openpose_coco.npy'
    batch_size = 1
    h_dim = 480
    w_dim = 480
    epochs = 10000

    img = tf.placeholder(tf.float32, [batch_size, h_dim, w_dim, 3])
    dropout = tf.placeholder_with_default(1.0, shape=())
    # human_3d_labels = tf.placeholder(tf.float32, [None, 14])
    # global_step = tf.Variable(0, trainable=False)

    # openpose net
    net = CmuNetwork({'image': img}, trainable=False)
    outputs = net.get_output()
    human_points, human_num = tf.py_func(post_est,
                                         [outputs, batch_size],
                                         [tf.float32, tf.int32],
                                         stateful=False, name='py_func')

    # linear net
    human_3d_predict = lineal_model(tf.reshape(human_points, [-1, 28]), dropout)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        net.load(pretrain_path, sess, False)
        tf.train.Saver().restore(sess,
                                 '/media/rodrigo/c1d7e9c9-c8cb-402e-b241-9090925389b3/human_action/test_tb/openpose_3d/save/save.ckpt')
        cap = cv2.VideoCapture(0)
        cap.set(3, w_dim)
        cap.set(4, h_dim)
        plt.ion()
        l1, l2 = [0, 1, 2, 3, 1, 5, 6, 1, 1, 8, 9, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 11, 9, 10, 12, 13]
        fig = plt.figure()
        af = fig.add_subplot(121)
        ax = fig.add_subplot(122, projection='3d')
        # ax.view_init(-90, -90)

        for epoch in range(epochs):

                frames = np.zeros([batch_size, w_dim, h_dim, 3])

                for fr in range(batch_size):
                    ret_val, image = cap.read()
                    padd_d = (image.shape[0] - image.shape[1]) // 2
                    d_1, d_2 = abs(padd_d), 0
                    image = np.pad(image, [[d_1, d_1], [d_2, d_2], [0, 0]], mode='constant')
                    image = cv2.resize(image, (h_dim, w_dim))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frames[fr] = image

                kp_3d, kp_2d = sess.run([human_3d_predict, human_points], {img: frames})
                kp_2d = kp_2d.reshape([batch_size, 14, 2])
                kp_3d = np.append(kp_2d, kp_3d.reshape(batch_size, 14, 1), axis=2)
                for fr in range(batch_size):
                    af.cla()
                    af.imshow(frames[fr].astype(np.uint8))

                    ax.cla()
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    # ax.view_init(-90, -90)
                    for ii in range(14):
                        xs, ys, zs = kp_3d[fr, ii]
                        if xs != 0:
                            ax.scatter(xs, ys, zs, c="r", marker="^")
                        for iii, pp in zip(l1, l2):
                            if kp_3d[fr, iii, 0] * kp_3d[fr, pp, 0] != 0:
                                ax.plot([kp_3d[fr, iii, 0], kp_3d[fr, pp, 0]],
                                        [kp_3d[fr, iii, 1], kp_3d[fr, pp, 1]],
                                        [kp_3d[fr, iii, 2], kp_3d[fr, pp, 2]],
                                        color='b')
                    plt.pause(0.01)
                    plt.draw()

    logger.info('Finish')

    print('finish')


if __name__ == '__main__':
    main()
