import pandas as pd
import numpy as np
import argparse
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class Dataset():
  def __init__(self, import_path, out_steps=1, need_normalization=False, rawData=False):
    self.df, self.plot_cols = self.import_data(import_path, rawData=rawData)   
    self.train_df, self.val_df, self.test_df, self.num_features = self.split(self.df) 
    if(need_normalization == True):
      self.train_df, self.val_df, self.test_df = self.normalize(self.df, self.train_df, self.val_df, self.test_df) 
    
    self.label_columns = self.df[self.df.columns[0]]
    if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(self.label_columns)}
    self.column_indices = {name: i for i, name in
                                enumerate(self.train_df.columns)}

    self.input_width = len(self.train_df)    # кількість вхідних даних
    self.label_width = self.input_width    # кількість позначень

    self.total_window_size = self.input_width + out_steps  # загальний розмір вікна даних як сума вхідних та зсуву (місця для прогнозу)

    self.input_indices = np.arange(self.total_window_size)[self.input_width]
    self.input_slice = slice(0, self.input_width)
    self.labels_slice = slice(self.input_width, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
  def __repr__(self):
    return '\n'.join([
      '\n',
      f'Розмірність вхідних даних: {self.df.shape}',
      f'Тренувальна вибірка: {self.train_df.shape}',
      f'Валідаційна вибірка: {self.val_df.shape}',
      f'Тестова вибірка: {self.test_df.shape}',
      f'Вхідні дані: {self.df}'])
    
  def import_data(self, data_path, sheet=0, rawData=False):
    if (rawData):
      df = import_path
      print(df)
    else:
      # df = pd.read_excel('Al.xlsx', sheet_name=1 , header = 0, usecols=[2,3])
      df = pd.read_excel('input.xlsx', sheet_name=4)
    # print('\nРозмірність вхідних даних: ', df.shape)
    # print('Вхідні дані:\n', df)

    # побудова графіку початкових даних
    fig = plt.figure(num="Початкові дані")
    plt.close()
    # timer = fig.canvas.new_timer(interval = 60000)
    # timer.add_callback(plt.close)
    plot_cols = df.columns[1:]
    plot_features = df[plot_cols]
    plot_features.index = df[df.columns[0]]
    _ = plot_features.plot(subplots=True)
    # timer.start()
    plt.show()
    df.describe().transpose()
    return df, plot_cols
  
  def split(self, df):
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)                             # загальна кількість елементів
    train_df = df[0:int(n*0.7)]             # 70% тренувальна вибірка
    val_df = df[int(n*0.7):int(n*0.9)]      # 20% для підтвердження результату
    test_df = df[int(n*0.9):]               # решта (10%) тестова 
    num_features = df.shape[1]
    return train_df, val_df, test_df, num_features
  
  def normalize(self, df, train_df, val_df, test_df):
    train_mean = train_df.mean()    
    train_std = train_df.std()      
    train_df = (train_df - train_mean) / train_std  
    val_df = (val_df - train_mean) / train_std      
    test_df = (test_df - train_mean) / train_std    
    
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Нормалізовано')
    plt.figure(figsize=(12, 6), num="Нормалізовані дані")
    ax = sns.violinplot(x='Column', y='Нормалізовано', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=0)
    # timer.start()
    plt.show()
    return train_df, val_df, test_df
  
  def get_dataset(self):
     return self.train_df, self.val_df, self.test_df, self.plot_cols  

  # звіт про склад вікна даних
  def __repr__(self):
    return '\n'.join([
      '\n',
      f'Загальний розмір вікна: {self.total_window_size}',
      f'Індекси вхідних даних: {self.input_indices}',
      f'Індекси міток даних: {self.label_indices}',
      # f'Назва стовпця(-ів): {self.label_columns}'
      ])
    
  # розподіл даних у вікні на значення та позначки
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)
    # встановлення значень стану
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels
  
  # функція для побудови графіку отриманих даних
  def plot(self, model=None, max_subplots=5):
    plot_col=self.plot_cols
    inputs, labels = self.example
    plt.figure(figsize=(10, 8))
    plot_col_index = self.column_indices[plot_col[0]]
    # plot_col_index = plot_col[0]
    # max_n = min(max_subplots, len(inputs))
    max_n = len(plot_col)
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col[n]}')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Вхідні дані', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Мітки', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        print('Inputs: ', inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Прогноз',
                    c='#ff7f0e', s=64)
      if n == 0:
        plt.legend()
    plt.xlabel('Ітерація експерименту')
    plt.show()
    
  # функція додавання даних до вікна даних (формування робочої вибірки)
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=10,)
    ds = ds.map(self.split_window)
    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
   
class Model(): 
 
  # Конструктор класу. Вхідні параметри: 
  # model_path: шлях до моделі (для імпорту готової, якщо є),
  # out_steps: кількість кроків прогнозу, 
  # num_features: кількість параметрів, 
  # verbose: детальність опису процесу навчання моделі (0 - нічого не виводиться)
  def __init__(self, model_path = None, 
               out_steps = 1, num_features = None, verbose=0):
    self.out_steps = out_steps
    self.num_features = num_features
    self.verbose = verbose
    if (model_path != None):
      self.model = tf.keras.models.load_model(model_path)
    elif (out_steps != None and num_features != None):
      self.model = self.create_model(out_steps, num_features)
  
  # Функція-репрезентація класу - дозволяє вивести ключові параметри класу для ознайомлення    
  def __repr__(self):
    return '\n'.join([
      '\n',
      'Тип моделі: Згорткова',      
      'Функція розрахунку втрат: MSE',
      'Оптимізатор: Adam',
      f'Кількість змінних для аналізу: {self.num_features}',
      f'Кількість прогнозованих кроків: {self.out_steps}'])
  
  # Метод створення моделі. Вхідні параметри: 
  # out_steps: кількість кроків прогнозу, 
  # num_features: кількість параметрів
  def create_model(self, out_steps, num_features):
      model = tf.keras.Sequential([                         # модель, що виконується послідовно
        tf.keras.layers.Lambda(lambda x: x[:, -3:, :]),         
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(3)),  # виконуємо для одного параметру виконуємо 1-вимірну згортку
        tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),  # шар ущільнення до [к-сть кроків прогнозу]*[к-сть параметрів] елементів
        tf.keras.layers.Reshape([OUT_STEPS, num_features])  
      ])
      return model

  def compile_and_fit(self, window, patience=2):
    model = self.model    
    # колбек, що дозволяє завершити навчання раніше при можливому погіршенні точності моделі (наприклад, при перенасиченості даними)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',               # відслідковування втрат при навчанні
                                                      patience=patience,            # чутливість до збільшення втрат (кількість епох, 
                                                                                    # втрати при навчанні яких більше за попередні, необхідна для переривання навчання)
                                                      # restore_best_weights=True,  # параметр, що повинен повертати модель до ваг, 
                                                                                    # що були найбільш вдалими, але позитивного ефекту не помітив
                                                      mode='min')                   # акцент на мінімум втрат
    
    model.compile(loss=tf.losses.MeanSquaredError(),        # втрати обчислюються як середньо-квадратична похибка
                  optimizer=tf.optimizers.Adam(),           # оптимізатор Adam. Детальніше: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
                  metrics=[tf.metrics.MeanAbsoluteError()]) # метрика базується на середньо-квадратичній похибці
  
  # Документація по методу: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    history = model.fit(window.train, epochs=MAX_EPOCHS,  # виконується тренування моделі на основі даних з [window.train]; результат записується у [history]; не більше [MAX_EPOCH] епох
                        validation_data=window.val,       # валідація виконується усередині [window.val]
                        verbose=self.verbose,             # детальність опису процесу навчання
                        callbacks=[early_stopping])       # можливі до застосування callback'и
    self.plot_train_history(history, 'Training and validation loss')    # побудова графіку втрат при навчанні
    return history
  
  # Метод побудови графіка втрат при навчанні.
  def plot_train_history(self, history, title):
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    # if(len(history.history['val_loss'])): 
    #   plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    
  def forecast():
    model.predict()
  


def arg():# завантаження датасета
  parser = argparse.ArgumentParser(description='Filter an xlsx-based data')
  parser.add_argument('-p','--path', type=str, default='input.xlsx',
                      help='Path to xlsx file with data.')
  parser.add_argument('-s','--sheet', type=int, nargs='+', default=[0],
                      help='Sheet number. By default using first sheet in file')
  args = parser.parse_args()
  return [args.path, args.sheet[0]]


# class Application():

MAX_EPOCHS = 100
TRAIN_COUNT = 16
OUT_STEPS = 2
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

# def __init__():
data_path, sheet = arg()

dataset = Dataset(data_path, OUT_STEPS)
print(repr(dataset))
model = Model(out_steps=OUT_STEPS, num_features=dataset.num_features)
print(repr(model))

# multi_window = WindowGenerator(
#                                 input_width=TRAIN_COUNT,            
#                                 label_width=OUT_STEPS,     
#                                 dataset=dataset.get_dataset(),
#                                 model=model.model)  
# print(repr(multi_window)    )

dataset.plot()   # побудова графіку вікна даних
# plt.show()

history = model.compile_and_fit(multi_window)

IPython.display.clear_output()
multi_val_performance = {}  
multi_performance = {}

# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)             # продуктивність відповідно до значень
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test)                  # продуктивність моделі
multi_window.plot(model.model)                                                     # побудова графіка отриманих результатів


# arch.save('convAlCold.h5')