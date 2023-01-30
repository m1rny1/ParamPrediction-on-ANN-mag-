import pandas as pd
import numpy as np
import argparse

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# налаштування кількості епох для тренування
MAX_EPOCHS = 400

# функція компіляції та запуску тренування моделі
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',   # відслідковування втрат при навчанні
                                                    patience=patience,    #
                                                    mode='min')           # акцент на мінімум втрат
  
  model.compile(loss=tf.losses.MeanSquaredError(),        # втрати обчислюються як середньо-квадратична похибка
                optimizer=tf.optimizers.Adam(),           #
                metrics=[tf.metrics.MeanAbsoluteError()]) # метрика базується на середньо-квадратичній похибці

  history = model.fit(window.train, epochs=MAX_EPOCHS,  # виконується тренування моделі на основі даних з [window.train]; результат записується у [history]; не більше [MAX_EPOCH] епох
                      validation_data=window.val,       # валідація виконується усередині [window.val]
                    #   verbose=0,
                      callbacks=[early_stopping])       # можливі до застосування callback'и
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

# завантаження датасета (в якості приклада використовується вибірка погодних умов, розподілених за часом)
parser = argparse.ArgumentParser(description='Filter an xlsx-based data')
parser.add_argument('-p','--path', type=str, default="input.xlsx",
                    help='Path to xlsx file with data.')
parser.add_argument('-s','--sheet', type=int, nargs='+', default=[0],
                    help='Sheet number. By default using first sheet in file')
args = parser.parse_args()

path    = args.path
sheet   = args.sheet[0]

# зчитування даних із xlsx-файлу
df = pd.read_excel(path, sheet_name=sheet , header = 0)
print("\nSamples on start::\n", df)

# побудова графіку початкових даних
fig = plt.figure(num="Початкові дані")
plt.close()
timer = fig.canvas.new_timer(interval = 60000)
timer.add_callback(plt.close)
# plot_cols = ['Подовження, %']
plot_cols = df.columns[1:] #['MISES 1', 'MISES 2', 'MISES 3', 'MISES 4', 'MISES 5']
plot_features = df[plot_cols]
plot_features.index = df[df.columns[0]]
_ = plot_features.plot(subplots=True)
timer.start()
plt.show()

# акуратизація даних
df.describe().transpose()

# швидке перетворення Фур'є для дискретизації значень температури та розподіл їх у часі
fft = tf.signal.rfft(df[plot_cols])
f_per_dataset = np.arange(0, len(fft))

# розділення даних на вибірки (70% тренувальна, 20% для підтвердження, 10% тестова)
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)                             # загальна кількість елементів
train_df = df[0:int(n*0.7)]             # 70% тренувальна вибірка
val_df = df[int(n*0.7):int(n*0.9)]      # 20% для підтвердження результату
test_df = df[int(n*0.9):]               # решта (10%) тестова

num_features = df.shape[1]

# виконання нормалізації даних
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
timer.start()
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
  def plot(self, model=None, plot_col=plot_cols[0], max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(8, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [норм.]')
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

w2 = WindowGenerator(85, 20, 20)
# об'єднання трьох зрізів (вибірок) для передачі у вікно даних
#  example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
#                            np.array(train_df[100:100+w2.total_window_size]),
#                            np.array(train_df[200:200+w2.total_window_size])])

example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size]),
                           np.array(train_df[400:400+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('Розміри подані як: (batch, time, features)')
print(f'Розміри вікна: {example_window.shape}')
print(f'Розміри вхідних даних: {example_inputs.shape}')
print(f'Розміри міток для перевірки: {example_labels.shape}')


# прогнозування декількох кроків
OUT_STEPS = 4    
multi_window = WindowGenerator(input_width=8,            # 24 вхідних кроки
                               label_width=OUT_STEPS,     # 24 позначки (= кількості прогнозованих для порівняння правильності прогнозу)
                               shift=OUT_STEPS)           # 24 прогнозованих кроки
for i in range(len(plot_cols)):
    multi_window.plot(plot_col=plot_cols[i])   # побудова графіку вікна даних
    plt.show()

# налаштування згорткової мережі для прогнозу
CONV_WIDTH = 3  # ширина згортки
multi_conv_model = tf.keras.Sequential([
    # Обгортка , що подає лямбда-вираз у вигляді шару [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Шар виконання 1-вимірної згортки => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Шар ущільнення => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Шар перетворення даних до заданого виду => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window) # компіляція моделі та її тренування на основі даних з [multi_window]

# очистка вікна виводу та створення змінних для аналізу продуктивності
IPython.display.clear_output()
multi_val_performance = {}  
multi_performance = {}

# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)             # продуктивність відповідно до значень
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)     # продуктивність моделі
for i in range(len(plot_cols)):
    multi_window.plot(multi_conv_model, plot_cols[i])                                                     # побудова графіка отриманих результатів
    # timer.start()
    plt.show()

