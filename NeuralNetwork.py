import numpy as np
import math
import os


class NeuralNetwork:
    def __init__(self):

        # Adjustable parameters
        self.learning_rate = 0.02
        self.lambda_value = 0.05

        # Weights- and weighted sum for each layer, number of input nodes
        self.weights = []
        self.w_sum = []
        self.inp_nodes = -1

        # Activation functions- and activations for each layer,
        # and available activation functions
        self.act_funcs = []
        self.activations = []
        self.act_dict = {
            'relu': lambda x: x * (x > 0),
            'relu_der': lambda x: x > 0,

            # Using stable softmax
            'softmax': lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) /
                                 np.exp(x - np.max(x, axis=1, keepdims=True)).sum(axis=1, keepdims=True),
            'softmax_der': lambda x: math.e ** x * (1 - math.e ** x)
        }

        # The loss used and available loss function(s)
        self.loss = 'quadratic'
        self.loss_dict = {
            'quadratic': lambda a, b: sum(0.5 * (a - b) ** 2),
            'quadratic_der': lambda a, b: a - b
        }

    def add_input_layer(self, nodes):
        self.inp_nodes = nodes

    # Adding a hidden layer with an activation function
    def add_hidden_layer(self, nodes, act_func='relu'):
        if len(self.weights) == 0:
            self.weights.append(np.random.uniform(low=-0.1, high=0.1, size=(nodes, self.inp_nodes)))
        else:
            self.weights.append(np.random.uniform(low=-0.1, high=0.1, size=(nodes, self.weights[-1].shape[0])))
        self.act_funcs.append(act_func)

    def set_loss(self, loss_func):
        self.loss = loss_func

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Training the NN for one epoch and returning loss and the predicted values
    def fit(self, x_train, y_train, x_val=None, y_val=None, batch_size=32, lr_decay=0.999):
        sum_loss = 0
        correct = 0

        # Looping through the training data and performing backpropagation on each batch
        for batch_num in range(math.ceil(x_train.shape[0] / batch_size)):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, x_train.shape[0])

            x_train_batch = x_train[batch_start:batch_end, :, :]
            y_train_batch = y_train[batch_start:batch_end, :, :]

            # Prediction and backpropagation for current batch
            y_pred_batch = self._feed_forward(x_train_batch)
            self._back_propagate(y_train_batch, y_pred_batch, batch_size)

            correct += np.count_nonzero((y_pred_batch.argmax(axis=1) - y_train_batch.argmax(axis=1)) == 0)
            sum_loss += self.calculate_loss(y_pred_batch, y_train_batch)[0] * (batch_end - batch_start)

            NeuralNetwork.print_progress(40, batch_end, x_train.shape[0], sum_loss, correct)

            self.learning_rate = self.learning_rate * lr_decay

        # Printing final output for epoch if validation data is given
        if x_val is not None:
            y_val_pred = self.predict(x_val)
            pred_loss = self.calculate_loss(y_val_pred, y_val)
            correct_val = np.count_nonzero((y_val_pred.argmax(axis=1) - y_val.argmax(axis=1)) == 0)

            NeuralNetwork.print_progress(40, x_train.shape[0], x_train.shape[0], sum_loss, correct,
                                         loss_val=pred_loss[0], correct_val=correct_val, val_samp=x_val.shape[0])
        print()
        return sum_loss / x_train.shape[0], correct

    # Making a prediction for a single sample
    def predict(self, x):
        return self._feed_forward(x)

    # Printing the progressbar, loss and number of correctly predicted (also for validation data if given)
    @staticmethod
    def print_progress(bars, batch_end, epoch_length, sum_loss, correct,
                       loss_val=None, correct_val=None, val_samp=None):

        r = int((batch_end / epoch_length) * bars)
        progressbar = '\r[' + ''.join('=' for _ in range(r)) + '>' + ''.join('-' for _ in range(bars - r)) + '] '
        progress = str(round(batch_end * 100 / epoch_length, 3)) + ' % (' + \
                   str(batch_end) + '/' + str(epoch_length) + ')'
        train_stats = '\tloss: ' + str(round(sum_loss / batch_end, 5)) + \
                      '\tcor: ' + str(correct) + '/' + str(batch_end) + \
                      ' (' + str(round(correct * 100 / batch_end, 4)) + ' %)'

        val_stats = ''
        if loss_val is not None:
            val_stats = '\tval_loss: ' + str(round(loss_val, 5)) + \
                        '\tval_cor: ' + str(correct_val) + '/' + str(val_samp) + \
                        ' (' + str(round(correct_val * 100 / val_samp, 4)) + ' %)'
        print(progressbar + progress + train_stats + val_stats, end='')

    # Feeding the input through the NN and returning the result
    def _feed_forward(self, x):
        self.activations = [x]
        self.w_sum = []
        temp_act = x

        # Iterating through the layers and propagating the activations
        for num, layer in enumerate(self.weights):
            self.w_sum.append(layer @ temp_act)  # weighted sum for layer
            temp_act = np.array(self.act_dict[self.act_funcs[num]](self.w_sum[-1]))  # activations for layer
            self.activations.append(np.array(temp_act))
        return self.activations[-1]

    # Loss function + l2 regularisation loss
    def calculate_loss(self, y_pred, y):
        return sum(self.loss_dict[self.loss](y_pred, y)) / y.shape[0] + \
               self.lambda_value * self.learning_rate * sum(map(lambda x: sum((x ** 2).flatten()), self.weights))

    # Back-propagating the errors
    def _back_propagate(self, y_target, y_pred, batch_size):
        weight_update = None
        for layer_num in range(-1, -len(self.weights) - 1, -1):
            # The layer is the output layer
            if layer_num == -1:
                activation_loss = self.loss_dict[self.loss + '_der'](y_pred, y_target)
            else:
                activation_loss = self.weights[layer_num + 1].transpose() @ sum_loss

                # Updating the weights with l2 reg. and back-prop in the previously calculated layer
                self.weights[layer_num + 1] = (1 - self.learning_rate * self.lambda_value) * \
                                              self.weights[layer_num + 1] + self.learning_rate * weight_update

            # Differentiation of activations in this layer
            activation_derivative = self.act_dict[self.act_funcs[layer_num] + '_der'](self.activations[layer_num])

            sum_loss = activation_loss * activation_derivative  # Hadamard product

            # Calculating the gradient
            weight_update = (sum_loss @ self.activations[layer_num - 1].transpose(0, 2, 1)).sum(axis=0) / batch_size

        # Updating the weights from the input layer
        self.weights[0] = (1 - self.learning_rate * self.lambda_value) * self.weights[0] + \
                          self.learning_rate * weight_update

    def load_weights(self, model_name):
        self.weights = []
        path = 'Weights/' + model_name + '/'
        for layer_name in os.listdir(path):
            self.weights.append(np.load(path + layer_name))
        self.inp_nodes = self.weights[0].shape[0]

    def save_weights(self, model_name):
        os.mkdir('Weights/' + model_name)
        for num, layer in enumerate(self.weights):
            np.save('Weights/' + model_name + '/layer_' + str(num), layer)
