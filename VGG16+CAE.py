#!/usr/bin/env python3.5
# -*-coding: utf-8 -*-
"""
Created on 2018-8-30
"""
import os
import numpy as np
import skimage.data
import skimage.transform
import skimage.color
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from PIL import Image
import utils

data_dir = './data_dir/'  # 样本数据存储的路径
log_dir = './log_dir/'  # 输出日志保存的路径
PLOT_DIR = './Out/plots'


# tensorboard --logdir "./log_dir"


#输出中间层卷积结果
def plot_conv_output(conv_img, name):   
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder

    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)
    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters

    num_filters = conv_img.shape[3]

    # get number of grid rows and columns

    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),

                             max([grid_r, grid_c]))

    # iterate filters

    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :, l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic')
        # remove any labels from the axes
        ax.set_xticks([])

        ax.set_yticks([])
        # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


class MyData:
    def __init__(self, file_name=None):
        self._where = int(300 * np.random.rand())
        self._images = []
        self._len = 0
        if file_name is not None:
            self.load_images_from(file_name)

    def load_images_from(self, images_dir):
        # 获得 images_dir中所有png图片在disk的位置
        # images_dir = 图片所在文件夹地址
        image_names = [os.path.join(images_dir, image_name)
                       for image_name in os.listdir(images_dir) if image_name.endswith(".png")]
        if len(image_names) == 0:
            print("File is empty! please check the file.")
            return None
        # 加载图片
        # image_names = [os.path.join("F:\Python\WorkPlace\Test", image_name)
        # for image_name in os.listdir("F:\Python\WorkPlace\Test") if image_name.endswith(".png")]
        # 加载所有图片到内存，注意内存是否足够大
        for f in image_names:
            tmp = skimage.data.imread(f)
            if tmp.shape != (224, 224, 3):
                # _image.append(skimage.transform.resize(tmp, (224, 224, 3)))
                # 下一句可注释
                # print('''Your picture's shape is {0}, it's should be (224,224,3), /
                # we would reshape it, but be carefull'''.format(tmp.shape))
                tmp = skimage.color.rgba2rgb(tmp)           #RGBA -> RGB
                self._images.append(skimage.transform.resize(tmp, (224, 224)))
            else:
                self._images.append(tmp)
        # 取消注释查看图片格式,RGB色彩值归一化至0-1
        # image = self._images[0]
        # print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
        # plt.imshow(image)
        # plt.show()
        self._len = len(self._images)
        pass

    def next_batch(self, batch_size):
        if self._where + batch_size > self._len:
            np.random.shuffle(self._images)
            self._where = 0
        data = self._images[self._where:self._where + batch_size]
        self._where += batch_size
        return data


def struct_graph():
    # 下面是VGG16部分网络结构
    # 定义输入和标签的占位符
    inputs_ = tf.placeholder(tf.float32, (None, 224, 224, 3), name="input")
    targets_ = tf.placeholder(tf.float32, (None, 224, 224, 3), name="target")
    learning_rate = tf.placeholder("float")

    ### 编码部分 Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 224x224x64
    tf.add_to_collection('conv_output', conv1)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 224x224x64
    tf.add_to_collection('conv_output', conv2)
    maxpool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 112x112x64
    conv3 = tf.layers.conv2d(inputs=maxpool1, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 112x112x128
    tf.add_to_collection('conv_output', conv3)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 112x112x128
    tf.add_to_collection('conv_output', conv4)
    maxpool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 56x56x128
    conv5 = tf.layers.conv2d(inputs=maxpool2, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 56x56x256

    conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 56x56x256

    conv7 = tf.layers.conv2d(inputs=conv6, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 56x56x256

    maxpool3 = tf.layers.max_pooling2d(conv7, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 28x28x256
    conv8 = tf.layers.conv2d(inputs=maxpool3, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv9 = tf.layers.conv2d(inputs=conv8, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(inputs=conv9, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 28x28x512
    maxpool4 = tf.layers.max_pooling2d(conv10, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 14x14x512
    conv11 = tf.layers.conv2d(inputs=maxpool4, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(inputs=conv11, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    maxpool5 = tf.layers.max_pooling2d(conv13, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 7x7x512
    # 去掉全连接层的平均池化
    encode = tf.layers.average_pooling2d(maxpool5, pool_size=(7, 7), strides=(1, 1), padding='same')
    # Now 1x1x512

    # Flatten the input data展平
    images_flat = tf.contrib.layers.flatten(encode)
    dense = tf.layers.dense(inputs=images_flat, units=784, activation=tf.nn.relu)
    # Now 1x784

    ### 解码部分(不属于vgg16 根据论文自己组合的) Decoder
    # 这部分我把 上采样与卷积 用 反卷积 替代，理论上二者没什么不同
    image_shaped_input = tf.reshape(dense, [-1, 7, 7, 16])

    # d_conv0 = tf.layers.conv2d_transpose(inputs=dense, filters=16, kernel_size=(3, 3),
    # strides=(1, 1), padding="same", activation=tf.nn.relu)
    # Now 7*7*16
    d_conv1 = tf.layers.conv2d_transpose(inputs=image_shaped_input, filters=32, kernel_size=(3, 3),
                                         strides=(2, 2), padding="same", activation=tf.nn.relu)
    tf.add_to_collection('conv_output', d_conv1)
    # Now 14*14*32
    d_conv2 = tf.layers.conv2d_transpose(inputs=d_conv1, filters=64, kernel_size=(3, 3),
                                         strides=(2, 2), padding="same", activation=tf.nn.relu)
    tf.add_to_collection('conv_output', d_conv2)
    # Now 28*28*64
    d_conv3 = tf.layers.conv2d_transpose(inputs=d_conv2, filters=128, kernel_size=(3, 3),
                                         strides=(2, 2), padding="same", activation=tf.nn.relu)
    tf.add_to_collection('conv_output', d_conv3)
    # Now 56*56*128
    d_conv4 = tf.layers.conv2d_transpose(inputs=d_conv3, filters=64, kernel_size=(3, 3),
                                         strides=(2, 2), padding="same", activation=tf.nn.relu)
    tf.add_to_collection('conv_output', d_conv4)
    # Now 112*112*64
    d_conv5 = tf.layers.conv2d_transpose(inputs=d_conv4, filters=3, kernel_size=(3, 3,),
                                         strides=(2, 2,), padding="same", activation=None)
    tf.add_to_collection('conv_output', d_conv5)
    # Now 224*224*3
    decoded = tf.nn.sigmoid(d_conv5)
    image_shaped_input = tf.reshape(decoded, [-1, 224, 224, 3])
    tf.summary.image('input', image_shaped_input, 10)
    # Define loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=d_conv5))
    tf.summary.scalar('loss', loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return loss, opt, inputs_, targets_, learning_rate, decoded


def train_model():
    mydata = MyData("D:\PyCharm 2018.2.1\workplace\stock\dataset\images\stock_image\\test")
    loss, opt, inputs_, targets_, learning_rate, decoded = struct_graph()
    with tf.Session() as sess:
        step = 50  # 训练步数
        epochs = 100  # 训练轮数
        batch_size = 5  # 每次训练的图片张数
        sess.run(tf.global_variables_initializer())
        m_saver = tf.train.Saver()
        mm = 0
        # 加载保存过的图和数据
        #m_saver = tf.train.import_meta_graph('D:\PyCharm 2018.2.1\workplace\\traffic\\testmodel\mnist_sss-15.meta')
        #m_saver.restore(sess, os.path.join("D:\PyCharm 2018.2.1\workplace\\traffic\\testmodel", 'mnist_sss-15'))
        m_saver.restore(sess, tf.train.latest_checkpoint("./testmodel/"))
        for e in range(epochs):
            for i in range(step):

                batch_images = mydata.next_batch(batch_size)

                # 用224的进行网络训练

                batch_loss, _ = sess.run([loss, opt], feed_dict={
                    learning_rate: 0.001, inputs_: batch_images, targets_: batch_images})
                # 输出每次训练得到的编码数值encode
                # images_flat0 = tf.contrib.layers.flatten(conv1)
                # encode_aim = sess.run(images_flat, feed_dict={inputs_: batch_images})
                # print("编码向量：\n", encode_aim)

                # 输出训练的每张图像

                ddecoded = sess.run(decoded, feed_dict={inputs_: batch_images, targets_: batch_images})
                plt.figure(figsize=(10, 10))
                for ii in range(5):
                    pic_matrix = np.array(batch_images[ii], dtype="float")
                    plt.subplot(5, 2, ii*2+1)
                    plt.axis('off')
                    plt.imshow(pic_matrix)

                    pic_matrix2 = np.array(ddecoded[ii], dtype="float")
                    plt.subplot(5, 2, ii * 2 + 2)
                    plt.axis('off')
                    plt.imshow(pic_matrix2)
                plt.show()


                print("step %d, training's batch_loss %g" % (i + 1, batch_loss))
                if (i % 10 == 0):
                    m_saver.save(sess, './testmodel/mnist_sss', global_step=mm)
                    mm = mm + 1


def load_model():
    mydata = MyData("D:\PyCharm 2018.2.1\workplace\\flower\\flower_photos\\tulips")
    # D:\PyCharm\2018.2.1\workplace\stock\dataset\images\stock_image
    loss, opt, inputs_, targets_, learning_rate, decoded = struct_graph()
    sess = tf.Session()
    m_saver = tf.train.Saver()

    # load the model
    # m_saver = tf.train.import_meta_graph('./model/mnist_slp-0.meta')
    m_saver.restore(sess, tf.train.latest_checkpoint("./CAEmodel/"))
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    for ii in range(3):
        batch_images = mydata.next_batch(2)
        # 测试
        summary, batch_loss, _ = sess.run([merged, loss, opt], feed_dict={
            learning_rate: 0.001, inputs_: batch_images, targets_: batch_images})
        train_writer.add_summary(summary, ii)
        ddecoded = sess.run(decoded[ii], feed_dict={inputs_: batch_images, targets_: batch_images})
        plt.subplot(2, 1, 1)
        # one_pic_arr = np.reshape(batch_images[ii], (224, 672))
        pic_matrix = np.array(batch_images[ii], dtype="float")
        # plt.imshow(pic_matrix)
        # pic_matrix0 = np.matrix(one_pic_arr, dtype="float")
        plt.imshow(pic_matrix)
        # one_pic_arr1 = np.reshape(ddecoded, (224, 672))
        pic_matrix2 = np.array(ddecoded, dtype="float")
        # pic_matrix3 = np.matrix(pic_matrix2, dtype="float")
        plt.subplot(2, 1, 2)
        # print(ddecoded)
        plt.imshow(pic_matrix2)
        plt.show()
        '''
        df = pd.DataFrame(pic_matrix2[1])
        df.to_csv('D:/vggCSV.csv')
        '''
        '''
        #
        r=Image.fromarray(ddecoded[:,:,0]).convert('L')
        g = Image.fromarray(ddecoded[:, :, 1])
        b = Image.fromarray(ddecoded[:, :, 2])
        T=Image.merge("RGB",(r,g,b))
        print(type(T),np.shape(T),np.shape(ddecoded))
        plt.imshow(T)
        plt.show()

        #print(batch_images[ii],decoded[ii])

        #plt.subplot(2, 1, 2)
        '''
        conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={inputs_: batch_images})

        for i, c in enumerate(conv_out[0]):
            plot_conv_output(c, 'conv{}'.format(i))
        print("step %d, training's batch_loss %g" % (ii + 1, batch_loss))
    train_writer.close()


def main():
    train_model()
    #load_model()


if __name__ == '__main__':
    main()
