#Siamese neural network

#install standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

#import tensorflow dependencies- functional api
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D , Dense , MaxPooling2D , Input , Flatten


#set GPU Growth
#Avoid out of memory error(OOM) by setting GPU Memory Groth Comsumption
#use this when youre rich enogh to afford a GPU

gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

#setup paths/directories
POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negetive')
ANC_PATH = os.path.join('data','anchor')

#make the directories
'''os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)'''

#move the downloaded labelled data into "data/negetive"

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw',directory)):
        EX_PATH=os.path.join('lfw', directory, file)
        NEW_PATH= os.path.join(NEG_PATH,file)
        os.replace(EX_PATH,NEW_PATH)

#access web cam and using open cv
#establish connection to the webcam
#UUID(Universally Unique Identifier) generates unique image names
import uuid
'''
cap = cv2.VideoCapture(0)#if 3 doesn't work try other number
while cap.isOpened():
    ret , frame =cap.read()

    #slicing the image pixels
    frame=frame[120:120+250,200:200+250,:]

    #collect anchors
    if cv2.waitKey(1) & 0XFF==  ord('a'): # hitting 'q' breaks
        imgname=os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))#create unique file path
        cv2.imwrite(imgname,frame)#write out anchor images

    #collect positive
    if cv2.waitKey(1) & 0XFF==  ord('p'): # hitting 'q' breaks
        imgname = os.path.join(POS_PATH, s'{}.jpg'.format(uuid.uuid1()))#crate unique file path
        cv2.imwrite(imgname,frame)#write out positve images


    #show image on screen
    cv2.imshow('Image Collection', frame)
    #wait for 1 millisec and check what's pressed on keyboard
    if cv2.waitKey(1) & 0XFF==  ord('q'): # hitting 'q' breaks
        break

cap.release()
cv2.destroyAllWindows()
'''
#using a generator to be able to loop through and grab all of  the files within the given specifc directory
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)# this means take everything in anchor folder which has jpg extention and take 300 of them
negetive=tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)

dir_test= anchor.as_numpy_iterator()
print(dir_test.next())

#preprocessing Scale and Resize

def preprocess(file_path):
    #read in image from file path
    byte_img = tf.io.read_file(file_path)
    #load the iamge
    img = tf.io.decode_jpeg(byte_img)
    #preprocessing
    img = tf.image.resize(img, (105,105))
    img = img/255.00
    return img

#Create lablled data
#(anchor,positive)==1,1,1,1,1
#(anchor,negetive)==0,0,0,0,0

positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))#prints/ or is an array of lablled 1s
negetives = tf.data.Dataset.zip((anchor,negetive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))#prints/or is an array of lablled 0s
data = positives.concatenate(negetives)#joining positive and negetive data together

#**Build train and test partition**

def twin_preprocess(input_img,validation_img,label):
    return(preprocess(input_img),preprocess(validation_img),label)

#Dataloader pipeline

data = data.map(twin_preprocess)#preprocessing the anchor and positive/negetive image both at once
data = data.cache()#caching the data
data = data.shuffle(buffer_size=1024)#mixing the data samples

#training partition
train_data = data.take(round(len(data)*.7)) #takes the data takes their length multiplies it by .7 which in 420
train_data = train_data.batch(16)# batching the data in group of 16
train_data = train_data.prefetch(8)# pipeling the 8 preftech images

#testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


#Build embedding layer

def make_embedding():
    inp = Input(shape=(105,105,3) , name = 'input_image')
    #1st block
    c1 = Conv2D(64 , (10,10), activation='relu')(inp)
    m2 = MaxPooling2D(64,(2,2), padding='same')(c1)

    #2md block
    c2 = Conv2D(128, (7,7), activation='relu')(m2)
    m2= MaxPooling2D(64, (2, 2), padding='same')(c2)

    #3rd block
    c3= Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    #last embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)



    return Model(inputs=[inp] , outputs=[d1] , name='embedding' )

embedding = make_embedding()
#embedding.summary()


#building a custom layer
#Siamese L1 DISTANCE CLASS
#main layer SIMILARITY CALCULATION
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)



l1=L1Dist()
#l1(input_embedding,validation_embedding)

#Make Siamese model
input_image = Input(name='input_img', shape=(105, 105, 3))
validation_image = Input(name='validation_img', shape=(105, 105, 3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)

siamese_layer = L1Dist()
distances = siamese_layer(inp_embedding,val_embedding)

classifier = Dense(1, activation='sigmoid')(distances)


def make_siamese_model():
    #Anchor inage input in the network
    input_image = Input(name='input_img', shape=(105,105,3))
    #Validation image in the network
    validation_image = Input(name='validation_img' , shape=(105,105,3))

    #Combine siamese distance componenets
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),embedding(validation_image))

    #Classification layer
    classifier = Dense(1,activation='sigmoid')(distances)
    return Model(inputs=[input_image,validation_image],outputs=classifier,name='Siamese_Network')

siamese_model = make_siamese_model()
siamese_model.summary()

#TRAINING
#Setup Loss and Optimizer


binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)#learning rate is 0.0001

#Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt , siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        #get anchor/positive image
        X = batch[:2]
        #Get label
        y = batch[2]

        #forward pass
        yhat = siamese_model(X, training= True)#making prediction
        #Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
    #calculate gradients
    grad = tape.gradient(loss,siamese_model.trainable_variables)

    #updated weights and use it on siamese model
    opt.apply_gradients(zip(grad,siamese_model.trainable_variables))

    return loss

#Training loop

def train(data, EPOCHS):
    #loop through epochs
    for epoch in range (1,EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        #loop tghrough each batch
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx+1)
        #Save Checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

#EPOCHS = 50
#train(train_data,EPOCHS)

#Import metric calculations

from tensorflow.keras.metrics import Recall , Precision
#get a batch of test data
test_input , test_val , y_true = test_data.as_numpy_iterator().next()

#making predictions

y_hat = siamese_model.predict([test_input , test_val])
print(y_hat)

#post processing the results
predictions = [1 if prediction > 0.5 else 0 for prediction in y_hat]
print(predictions)
print(y_true)


#creating a metric object
m = Recall()
#Calculating the recall value
m.update_state(y_true,y_hat)
#return recall result
print(m.result().numpy())

#creating a metric object
m = Precision()
#Calculating the recall value
m.update_state(y_true,y_hat)
#return recall result
print(m.result().numpy())

#Data Visualisation shit

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(test_input[0])
plt.subplot(1,2,2)
plt.imshow(test_val[0])
plt.show()

#Saving Model
#save weights
siamese_model.save('siamesemodel.h1')

#reload model
model = tf.keras.models.load_model('siamesemodel.h1' , custom_objects={'L1Dist': L1Dist , 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
model.predict([test_input , test_val])
print(model.predict([test_input , test_val]))


#Real Time Test

#verification function

def verify(model , detection_threshold , verifcation_threshold ):
    results = []
    for image in os.listdir(os.path.join('application_data' , 'verification_data')):
        input_img = preprocess(os.path.join('application_data', 'input_image' , 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_image' , image))

        result = model.predict(list(np.expand_dims([input_img , validation_img], axis=1)))
        results.append(result)

   #Detection Threshold = Metric above wihich a prediction is concidered positive
    detection = np.sum(np.array(results) > detection_threshold)
   #vefification Threshold = Proportion of positive prediction/Total positive samples
    verifcation = detection/len(os.listdir(os.path.join('application_data' , 'verification_data')))
    verified  = verifcation > verifcation_threshold

    return  results , verified

#Real Time prediction

cap = cv2.VideoCapture(0)
while cap.isOpened :
    ret , frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]

    cv2.imshow('Verification' , frame)

    #verify
    if cv2.waitKey(10) & 0xFF == ord('v'):
        #Save input image to input folder
        cv2.imwrite(os.path.join('application_data', 'input_image' , 'input_image.jpg') , frame)
        #Run Verification
        results , verified = verify(model , 0.9,0.7)
        print(verified)
    if cv2.waitKey(10) & 0xFF == ord('q') :
        break

cap.read()
cv2.destroyAllWindows()



