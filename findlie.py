import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import math
import random
import numpy as np

# LSTM model
class LieLSTM(nn.Module):
    def __init__(self, hidden_dim, input_size, gesture_size, seq_len):
        super(LieLSTM, self).__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_len
        self.gesture_size = gesture_size
       
        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=1 , batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim * seq_len, gesture_size)

    def forward(self, seq_batch_input):

        batch_size = list(seq_batch_input.size())[1]

        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)

        # run the lstm model
        lstm_out, (ht, ct) = self.lstm(seq_batch_input, hidden)

        #print(lstm_out.shape)
        #lstm_out = lstm_out[-1, :, :]

        # lstm_out = (seq_len, batch, hidden_size)
        # convert lstm_out to (batch_size, -1) to merge
        lstm_out = lstm_out.contiguous().view(batch_size, -1)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out)

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        #gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        #return gesture_probabilities
        return gesture_out

class RunModel():

    def __init__(self):
        basedir = os.path.abspath(os.path.dirname(__file__))

        number_of_features = 598
        number_of_hidden   = 32   # size of hidden layer
        number_of_gestures = 2    # output size
        sequence_length    = 20   # refers to the amount of timeframes to check
        STORAGE_PATH = "parameters.model"


        # TODO: GET EVAL TENSOR




        lstm_model = LieLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)

        lstm_model.load_state_dict(torch.load(STORAGE_PATH))

        # eval using the model
        lstm_model.eval()

        result = [0, 0]

        with torch.no_grad():
            tensor_size = eval_tensor.size()

            for i in range (0, tensor_size[0] - sequence_length):
                smol_tensor = eval_tensor[i:i+sequence_length,:]

                resulting_tensor = lstm_model(smol_tensor.view(sequence_length, 1, number_of_features).float())

                last_item = torch.argmax(resulting_tensor)

                result[last_item] += 1

        self.result = result

    def get_result(self):
        if self.result[0] > self.result[1]:
            return 0
        return 1