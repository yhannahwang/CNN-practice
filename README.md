# Building model by keras
 General steps:
 1. specify Architecture (node/layers define)
 2. Compile (loss functions/ optimization define)
 3. Fit (circle of back propagation, etc)
 4. predict
 
 1. Model specify
 # Import necessary modules
 import keras
 from keras.layers import Dense
 from keras.models import Sequential

 n_cols = predictors.shape[1]
 
 # Set up the model: model
 model = Sequential()
 
# Add the first layer
model.add(Dense(50,activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32,activation='relu'))


# Add the output layer
model.add(Dense(1))


 2. Compile (loss functions/ optimization define)
 two chioce:
 - specify the optimizer
   - controls the learning rate
   -'Adam' is usually a good choice
 - loss function
   - mse
 
 # Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
 
 3. Fit (circle of back propagation, etc)
 applying back propagation and gradient descent with data to update the weights
 (scaling data before fitting can ease optimization)
# Fit the model
model.fit(predictors,target)

## Classification
 loss function:'categorical_crossentropy'
simliar to log loss (lower is better)
 activation function: 'softmax'


# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)
# Set up the model
model = Sequential()
# Add the first layer
model.add(Dense(32,activation='relu',input_shape = (n_cols,)))
# Add the output layer
model.add(Dense(2,activation = 'softmax'))
# Compile the model
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',metrics=['accuracy'])
# Fit the model
model.fit(predictors,target)


## saving, reloading and using models

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)
 
## model optimization
- stochastic gradient descent
(dying neuron problem) caused by relu activation functions
- when choosing good learning rate:
# Import the SGD optimizer
from keras.optimizers import SGD 

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001,0.01,1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer,loss = 'categorical_crossentropy' )
    
    # Fit the model
    model.fit(predictors,target)


____________________________

##check model performance- model validation
# Fit the model
hist = model.fit(predictors,target, validation_split = 0.3)
   model validation
   
### avoid overfitting - early stopping  
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience= 2)
# Fit the model
model.fit(predictors,target,epochs = 30,validation_split=0.3, callbacks=[early_stopping_monitor])



_______________----
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation ='relu', input_shape=input_shape))
model_2.add(Dense(100, activation ='relu', input_shape=input_shape))

# Add the output layer
model_2.add(Dense(2,activation='softmax') )

# Compile model_2
model_2.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

model1
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22        
=================================================================
Total params: 242.0
Trainable params: 242
Non-trainable params: 0.0
_________________________________________________________________
None

model 2 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 100)                110       
_________________________________________________________________
dense_2 (Dense)              (None, 100)                110       
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22        
=================================================================
Total params: 242.0
Trainable params: 242
Non-trainable params: 0.0
_________________________________________________________________
None


________________
model capacity

increase layers and nodes until the mse not change
