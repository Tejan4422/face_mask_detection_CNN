from keras.layers import Input, Lambda, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

IMAGE_SIZE = [224,224]
train_path = 'T:/work/mask_detector/dataset/train'
test_path = 'T:/work/mask_detector/dataset/test'

#baseModel = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)
baseModel = Xception(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)
 
    
headModel = baseModel.output
headModel = MaxPooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
variable_for_classification= 2
headModel = Dense(variable_for_classification ,activation="softmax")(headModel)


for layer in baseModel.layers:
    layer.trainable = False

model = Model(inputs = baseModel.input, outputs = headModel)

model.summary()

model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('T:/work/mask_detector/dataset/train',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('T:/work/mask_detector/dataset/val',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

r = model.fit_generator(training_set,
                        steps_per_epoch = 110,
                        epochs = 5,
                        validation_data = test_set,
                        validation_steps = 233
                        )



#model.save('vgg_face_mask_detector.h5')
model.save('xception_face_detector.h5')


# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')












