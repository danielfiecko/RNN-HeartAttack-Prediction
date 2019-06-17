import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import load_model

def convert_dataset(dataset, window_size):
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size - 1):
        dataX.append(dataset[i:(i + window_size), 0])
        dataY.append(dataset[i + window_size, 0])
    return numpy.array(dataX), numpy.array(dataY)


def prepare_dataset(scaler, window_size):
    raw_dataframe = read_csv('PredictionDataSet-testset.csv',
                             usecols=[0], engine='python')
    raw_dataset = raw_dataframe.values.astype('float32')
    normalized_dataset = scaler.fit_transform(raw_dataset)
    bp_to_predict, bp_original = convert_dataset(normalized_dataset, window_size)
    # reshape input to be [samples, time steps, features]
    bp_to_predict = numpy.reshape(
        bp_to_predict, (bp_to_predict.shape[0], 1, bp_to_predict.shape[1]))
    return bp_to_predict, bp_original


def draw_results(bp_original, bp_predicted, scaler):
    bp_predicted_inversed = scaler.inverse_transform(bp_predicted)
    bp_original_inversed_values = scaler.inverse_transform([bp_original])[0]

    warmup_steps = int(len(bp_predicted_inversed) * 0.1)

    rmse_of_validation_set = math.sqrt(mean_squared_error(
        bp_original_inversed_values[warmup_steps:], bp_predicted_inversed[warmup_steps:, 0]))
    print('Validation RMSE: %.2f RMSE' % (rmse_of_validation_set))

    plt.style.use("seaborn")
    plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
    plt.plot(bp_original_inversed_values, color='grey', label='original')
    plt.plot(bp_predicted_inversed, color='yellowgreen', label='predicted')
    plt.title('PREDICTION OF SYSTOLIC BLOOD PRESSURE', fontweight='bold')
    plt.xlabel('SAMPLES')
    plt.ylabel('SBP')
    plt.legend()
    plt.show()


def main():
    window_size = 100
    scaler = MinMaxScaler(feature_range=(0, 1))
    bp_to_predict, bp_original = prepare_dataset(scaler, window_size)
    model = load_model('rnn_model.h5')
    bp_predicted = model.predict(bp_to_predict)
    draw_results(bp_original, bp_predicted, scaler)


main()
