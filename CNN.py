(train_X,train_Y), (test_X,test,Y) = fashion_mnist.load_data()

#FIND UNIQUE NUMBERS FROM TRAIN LABEL
 classes = np.unique(train_Y)
nClasses = len(classes)

##PREPROCESSING DATA

train_X = train_X.reshape(-1,28,28,1)
test_X = test_X.reshape(-1,28,28,1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

# CHANGE THE LABELS FROM CATEGORICAL TO ONE HOT ENCODING

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# split training data to trainig part and validation part
train_X,valid_X,train_label,valid_label = train_test_split(train_X,train_Y_one_hot,test_size = 0.2,random_state=-13)

batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential([
Conv2D(32,kernel_size=(3,3),activation=LeakyRelu(alpha=0.1),input_shape=(28,28,1),padding='same'),
MaxPooling2D((2,2),padding='same'),
Dropout(0.25),
Conv2D(64,(3,3),activation = LeakyRelu(alpha=0.1),padding='same'),
MaxPooling2D((2,2),padding='same'),
Dropout(0.25),
Conv2D(128,(3,3),activation = LeakyRelu(alpha=0.1),padding='same'),
MaxPooling2D((2,2),padding='same'),
Dropout(0.4),
Flatten(),
Dense(128,activation=LeakyRelu(alpha=0.1)),
Dropout(0.3)
Dense(num_classes,activation='softmax')
])


# COMPILE THE MODEL

class myCallBaclk(callback):
 def on_epoch_end(self,epochs,logs={}):
    if (logs.get('accuracy')>0.99):
       print('cancel')
       self.model.stop_training = True
callbacks = myCallBack()
fashion_model.compile = (loss = keras.losses.categorical_crossentropy,optimizer = keras.optimizers.Adam(),metrics=['Accuracy'])

# Train The Model

fashion_train = fashion_model.fit(train_X,train_label,batch_size=batch_size,epochs=epochs,verbose=-1,validation=(valid_X,valid_label),callbacks = [callbacks])

#SAVE_MODEL
model = keras.models.load_model("jhj")
y_pred = model.predict(Valid_X)