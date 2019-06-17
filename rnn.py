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


def convert_dataset(dataset, window_size):
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size - 1):
        dataX.append(dataset[i:(i + window_size), 0])
        dataY.append(dataset[i + window_size, 0])
    return numpy.array(dataX), numpy.array(dataY)


def prepare_dataset(scaler, window_size):
    raw_dataframe = read_csv('SBP-PredictionDataSet-training&validation.csv',
                             usecols=[0], engine='python')
    raw_dataset = raw_dataframe.values.astype('float32')

    normalized_dataset = scaler.fit_transform(raw_dataset)
    rows_amount = int(len(normalized_dataset))
    train_size = int(rows_amount * 0.9)
    validation_size = rows_amount - train_size
    train, validation = normalized_dataset[0:train_size,
                                           :], normalized_dataset[train_size: rows_amount, :]

    # reshape into X=t and Y=t+window_size
    trainX, trainY = convert_dataset(train, window_size)
    bp_to_validate, bp_original = convert_dataset(validation, window_size)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    bp_to_validate = numpy.reshape(
        bp_to_validate, (bp_to_validate.shape[0], 1, bp_to_validate.shape[1]))
    return bp_to_validate, bp_original, trainX, trainY


def draw_results(bp_original, bp_validated, scaler):
    bp_validated_inversed = scaler.inverse_transform(bp_validated)
    bp_original_inversed_values = scaler.inverse_transform([bp_original])[0]

    warmup_steps = int(len(bp_validated_inversed) * 0.1)

    rmse_of_validation_set = math.sqrt(mean_squared_error(
        bp_original_inversed_values[warmup_steps:], bp_validated_inversed[warmup_steps:, 0]))
    print('Validation RMSE: %.2f RMSE' % (rmse_of_validation_set))

    plt.style.use("seaborn")
    plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
    plt.plot(bp_original_inversed_values, color='grey', label='original')
    plt.plot(bp_validated_inversed, color='yellowgreen', label='validated')
    plt.title('VALIDATION OF SYSTOLIC BLOOD PRESSURE', fontweight='bold')
    plt.xlabel('SAMPLES')
    plt.ylabel('SBP')
    plt.legend()
    plt.show()


def main():
    window_size = 100
    scaler = MinMaxScaler(feature_range=(0, 1))
    bp_to_validate, bp_original, trainX, trainY = prepare_dataset(
        scaler, window_size)

    callback_early_stopping = EarlyStopping(
        monitor='loss', patience=10, verbose=1)
    callbacks = [callback_early_stopping]

    model = Sequential()
    model.add(LSTM(1, input_shape=(1, window_size),
                   activation='linear', return_sequences=True))
    model.add(LSTM(512, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, steps_per_epoch=62, epochs=100, batch_size=16,
              verbose=1, callbacks=callbacks)
    print(model.summary())
    model.save('rnn_model.h5')

    bp_validated = model.predict(bp_to_validate)
    draw_results(bp_original, bp_validated, scaler)


main()
