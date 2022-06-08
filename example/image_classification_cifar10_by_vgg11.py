# -*- coding: utf-8 -*-
# 使用VGG11在CIFAR-10上两个Epoch即可达到100%的准确率

from tensorflow.keras import datasets, optimizers, losses, utils, layers, models, callbacks
from cv4keras.model import VGG

num_classes = 10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
input_tensor = layers.Input(shape=(32, 32, 3))

vgg1 = VGG('vgg11', input_tensor=input_tensor, only_conv=True)
output = vgg1.outputs[0]

output = layers.Dense(num_classes, activation='softmax')(output)
output = layers.Flatten()(output)
model = models.Model(inputs=input_tensor, outputs=output)

model.summary(line_length=90)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=losses.categorical_crossentropy,
    metrics=['acc']
)


if __name__ == '__main__':

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=3,
        validation_data=(x_test, y_test),
        # callbacks=[callbacks.ModelCheckpoint('model.weights', save_best_only=True, save_weights_only=True)]
    )

    # with open('history.txt', 'w') as f:
    #     f.write(str(history.history))

else:
    model.load_weights('best_model.weights')
