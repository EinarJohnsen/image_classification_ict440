import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import matplotlib.pyplot as plt
import process_images as image_util
import weed_utils as wd
import numpy as np

FLAGS = None

def main():

    loss_values = []

    one_hot_encoded_labels_dummy = []
    one_hot_encoded_test_labels_dummy = []
    one_hot_encoded_verification_dummy = []
    filenames_dummy, labels_dummy, test_filenames_dummy, test_labels_dummy, verification_filenames_dummy, verification_labels_dummy = image_util.run_it()

    for x in labels_dummy:
        one_hot_encoded_labels_dummy.append(wd.one_hot_encoder2(x))

    for y in test_labels_dummy:
        one_hot_encoded_test_labels_dummy.append(wd.one_hot_encoder2(y))

    for z in verification_labels_dummy:
        one_hot_encoded_verification_dummy.append(wd.one_hot_encoder2(z))

    filenames = tf.constant(filenames_dummy)
    labels = tf.constant(one_hot_encoded_labels_dummy)
    test_filenames = tf.constant(test_filenames_dummy)
    test_lables = tf.constant(one_hot_encoded_test_labels_dummy)

    verification_filenames = tf.constant(verification_filenames_dummy)
    verification_lables = tf.constant(one_hot_encoded_verification_dummy)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(wd.parse_function)

    test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_filenames,test_lables))
    test_dataset = test_dataset.map(wd.parse_function)

    verification_dataset = tf.contrib.data.Dataset.from_tensor_slices((verification_filenames, verification_lables))
    verification_dataset = verification_dataset.map(wd.parse_function)

    dataset = dataset.batch(15)
    dataset = dataset.shuffle(len(filenames_dummy))
    dataset = dataset.repeat()

    test_dataset = test_dataset.batch(60)
    test_dataset = test_dataset.repeat()

    verification_dataset = verification_dataset.batch(30)
    verification_dataset = verification_dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()
    verification_iterator = verification_dataset.make_one_shot_iterator()

    # Show image and label (that it is correct)
    """
    with tf.Session() as sess:
        a,b = iterator.get_next()
        c,d = sess.run([a,b])
        print(c[99])
        print("----")
        print(d[99])
    """

    # Show a image
    """
    with tf.Session() as sess:
        a,b = iterator.get_next()
        c,d = sess.run([a,b])
        plt.imshow(c[99],)
        plt.show()
    """

    # PLACEHOLDERS
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 3])

    # W --> [filter H, filter W, Channels IN, Channels OUT]
    convo_1 = wd.convolutional_layer(x, shape=[5,5,3,32])
    # x --> [batch, h, w, c]
    convo_1_pooling = wd.max_pooling_2by2(convo_1)

    # W --> [filter H, filter W, Channels IN, Channels OUT]
    convo_2 = wd.convolutional_layer(convo_1_pooling, shape=[5,5,32, 32*2])
    # x --> [batch, h, w, c]
    convo_2_pooling = wd.max_pooling_2by2(convo_2)

    # W --> [filter H, filter W, Channels IN, Channels OUT]
    convo_2_flat = tf.reshape(convo_2_pooling, [-1,8*8*64])

    full_layer_one = tf.nn.relu(wd.normal_full_layer(convo_2_flat, 1024))
    full_layer_two = tf.nn.relu(wd.normal_full_layer(full_layer_one, 2048))
    full_layer_three = tf.nn.relu(wd.normal_full_layer(full_layer_two, 1024))
    full_layer_four = tf.nn.relu(wd.normal_full_layer(full_layer_three, 512))
    full_layer_five = tf.nn.relu(wd.normal_full_layer(full_layer_four, 1024))
    full_layer_six = tf.nn.relu(wd.normal_full_layer(full_layer_five, 512))

    # Dropout
    hold_prob = tf.placeholder(tf.float32)
    hold_prob_layer_two = tf.placeholder(tf.float32)
    hold_prob_layer_three = tf.placeholder(tf.float32)
    hold_prob_layer_four = tf.placeholder(tf.float32)
    hold_prob_layer_five = tf.placeholder(tf.float32)
    hold_prob_layer_six = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
    full_two_dropout = tf.nn.dropout(full_layer_two, keep_prob=hold_prob_layer_two)
    full_three_dropout = tf.nn.dropout(full_layer_three, keep_prob=hold_prob_layer_three)
    full_four_dropout = tf.nn.dropout(full_layer_four, keep_prob=hold_prob_layer_four)
    full_five_dropout = tf.nn.dropout(full_layer_five, keep_prob=hold_prob_layer_five)
    full_six_dropout = tf.nn.dropout(full_layer_six, keep_prob=hold_prob_layer_six)


    y_pred = wd.normal_full_layer(full_one_dropout, 3)

    # LOSS function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
    #train = optimizer.minimize(cross_entropy)

    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    tf.summary.scalar('cross entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summary_op = tf.summary.merge_all()
    
    '''
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
    '''

    init = tf.global_variables_initializer()

    steps = 

    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(steps):

            value_x, value_y = iterator.get_next()
            batch_x, batch_y = sess.run([value_x, value_y])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
            writer = tf.summary.FileWriter("Mono_Dino", sess.graph)
            #optimizeren = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
            _, data, cross_entropy_tester = sess.run([optimizer,
                summary_op, cross_entropy], feed_dict={
                    x: batch_x,
                    y_true: batch_y,
                    hold_prob: 0.4, 
                    #hold_prob_layer_two: 0.7,
                    #hold_prob_layer_three: 0.8,
                    #hold_prob_layer_four: 0.9,
                    #hold_prob_layer_five: 0.7,
                    #hold_prob_layer_six: 0.2
                })

            #if(i % 10 == 0):
            writer.add_summary(data, i)
            writer.flush()
            #loss_values.append(loss_val)
            #print(loss_val)
            print("Step: ", i)
            if i % 20 == 0:
                #print("ON STEP {}".format(i))
                #print("Accuracy: ")
                
                #print(test, i)
                writer = tf.summary.FileWriter("Mono_Dino_Test", sess.graph)
                test_1, test_2 = test_iterator.get_next()
                test_batch_x, test_batch_y = sess.run([test_1, test_2])
                test, cross_entropy_test ,test_accuracy, test_matches = sess.run(
                    [summary_op, cross_entropy, accuracy, matches],
                    feed_dict={
                        x: test_batch_x,
                        y_true: test_batch_y,
                        hold_prob: 1.0,
                        #hold_prob_layer_two: 1.0,
                        #hold_prob_layer_three: 1.0,
                        #hold_prob_layer_four: 1.0,
                        #hold_prob_layer_five: 1.0,
                        #hold_prob_layer_six: 1.0
                    })
                #print(test_matches, test_batch_y)
                print(test_accuracy )
                writer.add_summary(test, i)
                writer.flush()
        
        print("-----------------------")
        writer = tf.summary.FileWriter("Mono_Dino_Veri", sess.graph)
        verification_x, verification_y = verification_iterator.get_next()
        verification_batch_x, verification_batch_y = sess.run([verification_x, verification_y])
        test, verification_accuracy, verification_matches, test_y_pred, test_cross_ent = sess.run(
                    [summary_op, accuracy, matches, y_pred, cross_entropy],
                    feed_dict={
                        x: verification_batch_x,
                        y_true: verification_batch_y,
                        hold_prob: 1.0,
                        #hold_prob_layer_two: 1.0,
                        #hold_prob_layer_three: 1.0,
                        #hold_prob_layer_four: 1.0,
                        #hold_prob_layer_five: 1.0,
                        #hold_prob_layer_six: 1.0
                    })
        print(verification_batch_y, verification_matches, test_y_pred)
        print(verification_accuracy)
        print("-----------------------")
        writer.add_summary(test, i)
        writer.flush()
main()