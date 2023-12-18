#import relevant packages and libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers

#binarization function used to calculate gradients and binarize weights deterministically
@tf.custom_gradient
def binarize_d(weights):
    """
    Binarize weights with a custom gradient function.
    """
    def grad(upstream_grad):
        return tf.where(tf.abs(weights) < 1.0, upstream_grad, 0.0)
    weights_b = tf.where(weights >= 0, 1.0, -1.0)
    return weights_b, grad

#binarization function used to calculate gradients and binarize weights stochastically using hard sigmoid function
@tf.custom_gradient
def binarize_s(weights, stochastic_mask=None):
    """
    Binarize weights with a custom gradient function.
    """
    def grad(upstream_grad):
        return tf.where(tf.abs(weights) < 1.0, upstream_grad, 0.0)

    if stochastic_mask is None:
        p = tf.keras.activations.hard_sigmoid(weights)
        stochastic_mask = tf.where(tf.random.uniform(tf.shape(weights)) < p, 1.0, -1.0)
    
    return stochastic_mask, grad


def binarize_function(weights, method='deterministic'):
    # Choose between deterministic and stochastic binarization
    if method == 'deterministic':
        return binarize_d(weights)
    elif method == 'stochastic':
        return binarize_s(weights)
    else:
        raise ValueError('Invalid method specified. Choose "deterministic" or "stochastic".')

#add the constraint to the BinaryConnect class
class ClipWeights(Constraint):
    """Clips the weights incident to each layer to be between -1 and 1."""
    
    def __call__(self, w):
        return K.clip(w, -1.0, 1.0)

#a class to extent mlp layers with BinaryConnect algorithm 
class BinaryConnect(keras.layers.Layer):
    """
    BinaryConnect layer for binarizing weights with custom gradient.
    """

    def __init__(self, units, deterministic=True, kernel_constraint=None,kernel_regularizer=None, **kwargs):
        super(BinaryConnect, self).__init__(**kwargs)
        self.units = units
        self.deterministic = deterministic
        self.kernel_constraint = keras.constraints.get(kernel_constraint)  # Store the kernel_constraint
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.binarize_function = binarize_d if deterministic else binarize_s

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
            constraint=self.kernel_constraint,  # Apply the kernel_constraint
            regularizer=self.kernel_regularizer
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )

        self.real_weights = tf.Variable(self.kernel.read_value(), trainable=False)

        super(BinaryConnect, self).build(input_shape)  # Ensure proper build

    #function used for test and inference, use real weights for stochastic binarization
    def use_real_weights(self):
        if hasattr(self, 'real_weights'):
            self.kernel.assign(self.real_weights)

    #function used for test and inference, use binarized weights for deterministic binarization
    def use_binarized_weights(self):
        if hasattr(self, 'real_weights'):
            binarized_weights = self.binarize_function(self.real_weights)
            self.kernel.assign(binarized_weights)

    def call(self, inputs):
        if self.deterministic:
            weights_b = binarize_d(self.kernel)
        else:
            weights_b = binarize_s(self.kernel)
        outputs = tf.matmul(inputs, weights_b) + self.bias
        return outputs

#used in MLP model for the MNIST dataset
class BinaryConnectMLP(keras.Model):
    """
    MLP model with BinaryConnect layers.
    """

    def __init__(self, deterministic=True):
        super(BinaryConnectMLP, self).__init__()
        #reg = regularizers.l2(0.001)
        self.deterministic = deterministic
        self.bn1 = keras.layers.BatchNormalization()
        #self.dense1 = BinaryConnect(1024, deterministic=self.deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.dense1 = BinaryConnect(1024, deterministic=self.deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=None)
        self.relu1 = keras.layers.ReLU()
        self.bn2 = keras.layers.BatchNormalization()
        self.dense2 = BinaryConnect(1024, deterministic=self.deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=None)
        self.relu2 = keras.layers.ReLU()
        self.bn3 = keras.layers.BatchNormalization()
        self.dense3 = BinaryConnect(1024, deterministic=self.deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=None)
        self.relu3 = keras.layers.ReLU()
        self.dense4 = keras.layers.Dense(10, activation="softmax")
        #self.dense4 = keras.layers.Dense(10)

    #function used for test and inference, use real weights for stochastic binarization
    def use_real_weights(self):
        if hasattr(self, 'real_weights'):
            self.kernel.assign(self.real_weights)

    #function used for test and inference, use binarized weights for deterministic binarization
    def use_binarized_weights(self):
        if hasattr(self, 'real_weights'):
            binarized_weights = self.binarize_function(self.real_weights)
            self.kernel.assign(binarized_weights)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.bn3(x)
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.dense4(x)
        return x

#function to create the model used in MNIST dataset
def create_model(deterministic):
    model = BinaryConnectMLP(deterministic=deterministic)
    model.compile(
    #optimizer=keras.optimizers.legacy.SGD(learning_rate=0.01, decay=0.05), loss=tf.keras.losses.CategoricalHinge(), metrics=['accuracy'],)
    #optimizer='Adam', loss=tf.keras.losses.CategoricalHinge(), metrics=['accuracy'],)
    optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'],)
    return model


class BinaryConnectConv(keras.layers.Layer):
    """
    CNN model with BinaryConnect layers.
    """
    def __init__(self, filters, kernel_size, deterministic=True, kernel_constraint=None,kernel_regularizer=None,**kwargs):
        super(BinaryConnectConv, self).__init__(**kwargs)
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = regularizers.get(kernel_regularizer)  # Get the regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.deterministic = deterministic
        # Define binarize_function based on deterministic attribute
        self.binarize_function = binarize_d if self.deterministic else binarize_s

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", 
            shape=[self.kernel_size, self.kernel_size, input_shape[-1], self.filters],
            initializer="glorot_uniform",
            trainable=True,    
            constraint=self.kernel_constraint,  # Apply the kernel_constraint
            regularizer=self.kernel_regularizer  # Apply the kernel_regularizer
        )
        
        # Initialize real_weights after applying the constraint
        self.real_weights = tf.Variable(self.kernel.read_value(), trainable=False)



    #function used for test and inference, use real weights for stochastic binarization
    def use_real_weights(self):
        if hasattr(self, 'real_weights'):
            self.kernel.assign(self.real_weights)

    #function used for test and inference, use binarized weights for deterministic binarization
    def use_binarized_weights(self):
        if hasattr(self, 'real_weights'):
            binarized_weights = self.binarize_function(self.real_weights)
            self.kernel.assign(binarized_weights)

    def call(self, inputs):
        # Binarize the kernel using the appropriate function
        binarized_kernel = binarize_d(self.kernel) if self.deterministic else binarize_s(self.kernel)
        
        # Perform the convolution using the binarized kernel
        return tf.nn.conv2d(inputs, binarized_kernel, strides=[1, 1, 1, 1], padding="SAME")

""" BinaryConnect CNN model with the same architecture as described by the paper, used in the beginning of the project, adapt to the BinaryConnectCNN_SVHN model
class architecture for improving model results"""
class BinaryConnectCNN(keras.Model):
    def __init__(self, deterministic=True):
        super(BinaryConnectCNN, self).__init__()
        # Example: Add L2 regularization with a factor of 0.01
        reg = regularizers.l2(0.001)
        self.conv1_1 = BinaryConnectConv(128, kernel_size=3, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn1_1 = layers.BatchNormalization()
        self.conv1_2 = BinaryConnectConv(128, kernel_size=3, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn1_2 = layers.BatchNormalization()
        self.mp1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv2_1 = BinaryConnectConv(256, kernel_size=3, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn2_1 = layers.BatchNormalization()
        self.conv2_2 = BinaryConnectConv(256, kernel_size=3, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn2_2 = layers.BatchNormalization()
        self.mp2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.conv3_1 = BinaryConnectConv(512, kernel_size=3, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn3_1 = layers.BatchNormalization()
        self.conv3_2 = BinaryConnectConv(512, kernel_size=3, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn3_2 = layers.BatchNormalization()
        self.mp3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.fc1 = BinaryConnect(1024, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn_fc1 = layers.BatchNormalization()
        self.fc2 = BinaryConnect(1024, deterministic=deterministic,kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn_fc2 = layers.BatchNormalization()
        
        self.output_layer = keras.layers.Dense(10, activation='softmax')  # output layer, paper used l2-svm layer but I used softmax throughout the project
        #self.output_layer = keras.layers.Dense(10)

    #function used for test and inference, use real weights for stochastic binarization
    def use_real_weights(self):
        if hasattr(self, 'real_weights'):
            self.kernel.assign(self.real_weights)

    #function used for test and inference, use binarized weights for deterministic binarization
    def use_binarized_weights(self):
        if hasattr(self, 'real_weights'):
            binarized_weights = self.binarize_function(self.real_weights)
            self.kernel.assign(binarized_weights)

    def call(self, inputs):
        # First block
        x = self.conv1_1(inputs)
        x = self.bn1_1(x)
        x = tf.nn.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = tf.nn.relu(x)
        x = self.mp1(x)

        # Second block
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = tf.nn.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = tf.nn.relu(x)
        x = self.mp2(x)

        # Third block
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = tf.nn.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = tf.nn.relu(x)
        x = self.mp3(x)

        # Fully connected layers
        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = tf.nn.relu(x)

        return self.output_layer(x)


"""model class similar to previous BinaryConnectCnn class, this is the final version of the model architecture used in the project for both 
cifar-10 and SVHN dataset, applied dropout layers and L2 regularizer to prevent overfitting
"""
class BinaryConnectCNN_SVHN(keras.Model):
    def __init__(self, deterministic=True):
        super(BinaryConnectCNN_SVHN, self).__init__()
        reg = regularizers.l2(0.005)  # Regularization factor
        self.conv1_1 = BinaryConnectConv(256, kernel_size=3, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn1_1 = layers.BatchNormalization()
        self.conv1_2 = BinaryConnectConv(256, kernel_size=3, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn1_2 = layers.BatchNormalization()
        self.mp1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(0.1)

        self.conv2_1 = BinaryConnectConv(512, kernel_size=3, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn2_1 = layers.BatchNormalization()
        self.conv2_2 = BinaryConnectConv(512, kernel_size=3, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn2_2 = layers.BatchNormalization()
        self.mp2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = layers.Dropout(0.1)

        self.conv3_1 = BinaryConnectConv(1024, kernel_size=3, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn3_1 = layers.BatchNormalization()
        self.conv3_2 = BinaryConnectConv(1024, kernel_size=3, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn3_2 = layers.BatchNormalization()
        self.mp3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout3 = layers.Dropout(0.2)

        self.fc1 = BinaryConnect(2048, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn_fc1 = layers.BatchNormalization()
        self.dropout_fc1 = layers.Dropout(0.3)
        self.fc2 = BinaryConnect(2048, deterministic=deterministic, kernel_constraint=ClipWeights(), kernel_regularizer=reg)
        self.bn_fc2 = layers.BatchNormalization()
        self.dropout_fc2 = layers.Dropout(0.3)
        
        self.output_layer = layers.Dense(10, activation='softmax')

    #function used for test and inference, use real weights for stochastic binarization
    def use_real_weights(self):
        if hasattr(self, 'real_weights'):
            self.kernel.assign(self.real_weights)

    #function used for test and inference, use binarized weights for deterministic binarization
    def use_binarized_weights(self):
        if hasattr(self, 'real_weights'):
            binarized_weights = self.binarize_function(self.real_weights)
            self.kernel.assign(binarized_weights)

    def call(self, inputs):
        # First block
        x = self.conv1_1(inputs)
        x = self.bn1_1(x)
        x = tf.nn.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = tf.nn.relu(x)
        x = self.mp1(x)
        x = self.dropout1(x)

        # Second block
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = tf.nn.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = tf.nn.relu(x)
        x = self.mp2(x)
        x = self.dropout2(x)

        # Third block
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = tf.nn.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = tf.nn.relu(x)
        x = self.mp3(x)
        x = self.dropout3(x)

        # Fully connected layers
        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = tf.nn.relu(x)
        x = self.dropout_fc2(x)

        return self.output_layer(x)
    
"""a custom training function with a component to evaluate the model based on the type of binarization, used binarzied weights for deterministic binarization
and real valued weights for stochastical binarization as mentioned in the paper, used initially but later adapted to a more sophisticated training function
called custum_train_cnn through out the project"""    

def custom_train(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    # Prepare the dataset in batches
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    # Training loop
    for epoch in range(epochs):
        # Initialize metrics
        train_loss = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy()
        val_loss = keras.metrics.Mean()
        val_accuracy = keras.metrics.CategoricalAccuracy()

        # Training phase
        for x_batch_train, y_batch_train in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch_train, training=True)
                loss_value = keras.losses.categorical_crossentropy(y_batch_train, predictions)
            grads = tape.gradient(loss_value, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update real_weights here
            for layer in model.layers:
                if isinstance(layer, (BinaryConnect, BinaryConnectConv)):
                    layer.real_weights.assign(layer.kernel.read_value())

            # Update training metrics
            train_loss(loss_value)
            train_accuracy(y_batch_train, predictions)

        # Validation phase
        for x_batch_val, y_batch_val in val_dataset:
            # Use binarized weights for deterministic, real weights for stochastic
            for layer in model.layers:
                if isinstance(layer, (BinaryConnect, BinaryConnectConv)):
                    if layer.deterministic:
                        layer.use_binarized_weights()
                    else:
                        layer.use_real_weights()

            val_logits = model(x_batch_val, training=False)
            v_loss = keras.losses.categorical_crossentropy(y_batch_val, val_logits)

            # Update validation metrics
            val_loss(v_loss)
            val_accuracy(y_batch_val, val_logits)

            # Revert to real weights for the next training iteration
            for layer in model.layers:
                if isinstance(layer, (BinaryConnect, BinaryConnectConv)):
                    layer.use_real_weights()

        # Print training and validation results
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss.result()}, Train Accuracy: {train_accuracy.result()*100}, '
              f'Val Loss: {val_loss.result()}, Val Accuracy: {val_accuracy.result()*100}')

    return model

"""adapted version of custom_train function, this is the final training function used in the project,
adding a component of shuffling the dataset to improve model's ability to generalize to unseen data and reduce the 
influence of data orders in the training. store the relevant training loss which can be used for plotting also adding a functionality of
early stopping to reduce training time"""

def custom_train_cnn(model, x_train, y_train, x_val, y_val, epochs, batch_size, early_stopping_patience):
    # Prepare the dataset in batches
    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Initialize lists to store metrics
    epoch_train_losses = []
    epoch_val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Shuffle the training data at the beginning of each epoch
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

        # Initialize metrics
        train_loss = keras.metrics.Mean()
        train_accuracy = keras.metrics.CategoricalAccuracy()
        val_loss = keras.metrics.Mean()
        val_accuracy = keras.metrics.CategoricalAccuracy()

        # Training phase
        for x_batch_train, y_batch_train in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch_train, training=True)
                loss_value = keras.losses.categorical_crossentropy(y_batch_train, predictions)
            grads = tape.gradient(loss_value, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update real_weights here
            for layer in model.layers:
                if isinstance(layer, (BinaryConnect, BinaryConnectConv)):
                    layer.real_weights.assign(layer.kernel.read_value())

            # Update training metrics
            train_loss(loss_value)
            train_accuracy(y_batch_train, predictions)

        # Validation phase
        for x_batch_val, y_batch_val in val_dataset:
            # Use binarized weights for deterministic, real weights for stochastic
            for layer in model.layers:
                if isinstance(layer, (BinaryConnect, BinaryConnectConv)):
                    if layer.deterministic:
                        layer.use_binarized_weights()
                    else:
                        layer.use_real_weights()

            val_logits = model(x_batch_val, training=False)
            v_loss = keras.losses.categorical_crossentropy(y_batch_val, val_logits)

            # Update validation metrics
            val_loss(v_loss)
            val_accuracy(y_batch_val, val_logits)

            # Revert to real weights for the next training iteration
            for layer in model.layers:
                if isinstance(layer, (BinaryConnect, BinaryConnectConv)):
                    layer.use_real_weights()


        current_val_loss = val_loss.result().numpy()
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:

            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        # Store metrics at the end of each epoch

        epoch_train_losses.append(train_loss.result().numpy())
        epoch_val_accuracies.append(val_accuracy.result().numpy())

        # Print training and validation results
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss.result()}, Train Accuracy: {train_accuracy.result()*100}, '
              f'Val Loss: {val_loss.result()}, Val Accuracy: {val_accuracy.result()*100}')

    # Return the stored metrics
    return epoch_train_losses, epoch_val_accuracies