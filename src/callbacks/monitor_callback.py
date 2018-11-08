import tensorflow as tf


class MonitorCallBack(tf.keras.callbacks.Callback):
    def __init__(self, model, test_generator, args):
        super(MonitorCallBack, self).__init__()
        self.model = model
        self.test_generator = test_generator
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        # dummy_x = tf.zeros((1, 128, 157, 2))
        # self.model._set_inputs(dummy_x, training=False)
        scores = self.model.evaluate_generator(generator=self.test_generator, steps=5036 // self.args.batch_size)
        print("Final test loss:{0:.6f} accuracy:{1:.2f}".format(scores[0], scores[1]))

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass
