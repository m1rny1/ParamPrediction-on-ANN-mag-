import pandas as pd
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# налаштування кількості епох для тренування
MAX_EPOCHS = 100
TRAIN_COUNT = 30
OUT_STEPS = 2

# функція компіляції та запуску тренування моделі
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',   # відслідковування втрат при навчанні
                                                    patience=patience,
                                                    # restore_best_weights=True,
                                                    mode='min')           # акцент на мінімум втрат
  
  model.compile(loss=tf.losses.MeanSquaredError(),        # втрати обчислюються як середньо-квадратична похибка
                optimizer=tf.optimizers.Adam(),           #
                metrics=[tf.metrics.MeanAbsoluteError()]) # метрика базується на середньо-квадратичній похибці

  history = model.fit(window.train, epochs=MAX_EPOCHS,  # виконується тренування моделі на основі даних з [window.train]; результат записується у [history]; не більше [MAX_EPOCH] епох
                      validation_data=window.val)       # валідація виконується усередині [window.val]
                      # verbose=1,
                      # callbacks=[early_stopping])       # можливі до застосування callback'и
  return history
  """Compile and trains selected model in data window

  Returns:
      History.history : a record of training loss values and metrics values
    at successive epochs, as well as validation loss values
    and validation metrics values (if applicable).
  """

# нвлаштування вікна відображення графіків
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

def arg():# завантаження датасета
  parser = argparse.ArgumentParser(description='Filter an xlsx-based data')
  parser.add_argument('-p','--path', type=str, default='Al.xlsx',
                      help='Path to xlsx file with data.')
  parser.add_argument('-s','--sheet', type=int, nargs='+', default=[1],
                      help='Sheet number. By default using first sheet in file')
  args = parser.parse_args()
path    = 'Al.xlsx' # args.path
sheet   = 1         # args.sheet[0]


df = pd.read_excel(path, sheet_name=sheet , header = 0, usecols=[2,3])
print("\nSamples on start::\n", df)


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

# формування описової статистики
df.describe().transpose()

# розділення даних на вибірки (70% тренувальна, 20% для підтвердження, 10% тестова)
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)                             # загальна кількість елементів
train_df = df[0:int(n*0.7)]             # 70% тренувальна вибірка
val_df = df[int(n*0.7):int(n*0.9)]      # 20% для підтвердження результату
test_df = df[int(n*0.9):]               # решта (10%) тестова 
num_features = df.shape[1]
print("Тренувальна вибірка: ", train_df.shape)
print("Валідаційна вибірка: ", val_df.shape)
print("Тестова вибірка: ", test_df.shape)



train_mean = train_df.mean()    # тренувальне середнє значення
train_std = train_df.std()      # тренувальне стандартизоване
train_df = (train_df - train_mean) / train_std  # нормалізована тренувальна вибірка
val_df = (val_df - train_mean) / train_std      # нормалізована вибірка для підтвердження
test_df = (test_df - train_mean) / train_std    # нормалізована тестова вибірка
# побудова графіку нормалізованих даних
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Нормалізовано')
plt.figure(figsize=(12, 6), num="Нормалізовані дані")
ax = sns.violinplot(x='Column', y='Нормалізовано', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=0)
# timer.start()
plt.show()

# клас, що описує вікно досліджуваних даних
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # містять необроблені дані відповідних вибірок
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # налаштування параметрів вікна даних
    self.input_width = input_width    # кількість вхідних даних
    self.label_width = label_width    # кількість позначень
    self.shift = shift                # зсув даних

    self.total_window_size = input_width + shift  # загальний розмір вікна даних як сума вхідних та зсуву (місця для прогнозу)

    self.input_slice = slice(0, input_width)      # виокремлення зрізу відповідно до кількості вхідних даних
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  # звіт про склад вікна даних
  def __repr__(self):
    return '\n'.join([
        f'Загальний розмір вікна: {self.total_window_size}',
        f'Кількість вхідних даних: {self.input_indices}',
        f'Кількість міток даних: {self.label_indices}',
        f'Назва стовпця(-ів): {self.label_columns}'])
    
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
  def plot(self, model=None, plot_col=plot_cols, max_subplots=5):
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
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Прогноз',
                    c='#ff7f0e', s=64)
      if n == 0:
        plt.legend()
    plt.xlabel('Ітерація експерименту')
    
  # функція додавання даних до вікна даних (формування робочої вибірки)
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)
    ds = ds.map(self.split_window)
    return ds

  # використання декоратора property() для щвидкого доступу до створення відповідних вибірок
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

# прогнозування декількох кроків
multi_window = WindowGenerator(
                                input_width=TRAIN_COUNT,            # вхідних кроки
                                label_width=OUT_STEPS,     # позначки (= кількості прогнозованих для порівняння правильності прогнозу)
                                shift=OUT_STEPS)           # прогнозованих кроки
# multi_inputs, multi_labels = multi_window.split_window(multi_window)

# print('Розміри подані як: (batch, time, features)')
# print(f'Розміри вікна: {multi_window.shape}')
# print(f'Розміри вхідних даних: {multi_window.input_width.shape}')
# print(f'Розміри міток для перевірки: {multi_window.label_indices.shape}')

# for i in range(len(plot_cols)):
multi_window.plot(plot_col=plot_cols)   # побудова графіку вікна даних
plt.show()

# налаштування згорткової мережі для прогнозу
CONV_WIDTH = 3  # ширина згортки
multi_conv_model = tf.keras.Sequential([

    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),     # Обгортка , що подає лямбда-вираз у вигляді шару [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),   # Шар виконання 1-вимірної згортки => [batch, 1, conv_units] 
    tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()), # Шар ущільнення => [batch, 1,  out_steps*features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])  # Шар перетворення даних до заданого виду => [batch, out_steps, features]
])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(200, activation='relu', input_shape=(TRAIN_COUNT, OUT_STEPS)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(OUT_STEPS)
])

history = compile_and_fit(model, multi_window) # компіляція моделі та її тренування на основі даних з [multi_window]


def plot_train_history(history, title):
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

plot_train_history(history, 'Training and validation loss')

# очистка вікна виводу та створення змінних для аналізу продуктивності
IPython.display.clear_output()
multi_val_performance = {}  
multi_performance = {}

# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)             # продуктивність відповідно до значень
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test)                  # продуктивність моделі
multi_window.plot(multi_conv_model, plot_cols)                                                     # побудова графіка отриманих результатів
plt.show()

