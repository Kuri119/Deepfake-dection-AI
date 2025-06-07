from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model


train = ImageDataGenerator(rescale = 1./255)
valid = ImageDataGenerator(rescale = 1./255)

path = "./real-vs-fake/"

train_dataset = train.flow_from_directory(
    path + "train/",
    target_size = (224,224),
    batch_size = 100,
    class_mode = 'binary'
)

print(train_dataset.class_indices)

valid_dataset = valid.flow_from_directory(
    path + "valid/",
    target_size = (224,224),
    batch_size = 100,
    class_mode = 'binary'
)

print(valid_dataset.class_indices)


model = Sequential()

model.add(Conv2D(filters= 16, kernel_size=(3,3), activation='relu', input_shape= (224,224,3), padding = 'same'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters= 32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters= 64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

history = model.fit(train_dataset, steps_per_epoch = 100000//100, validation_data = valid_dataset, validation_steps = 20000//100, epochs = 10)

model.save('deepfake_3conv_16-32-64_dense_1024.keras')
