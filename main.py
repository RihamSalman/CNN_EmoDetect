from aiogram import Bot, Dispatcher, executor, types
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
num_words = 20000
max_messages_len = 50
nb_classes = 6
train = pd.read_csv('train.csv',
                    sep = ';',
                    header=None, 
                    names=['text', 'class'])

messages = train['text']
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(messages)
tokenizer.word_index
loaded_model = load_model('best_model_cnn.h5')

def show_result(predicted_value):
    if predicted_value == 0:
        return 'sadness'
    if predicted_value == 1:
        return 'joy'
    if predicted_value == 2:
        return 'love'
    if predicted_value == 3:
        return 'anger'
    if predicted_value == 4:
        return 'fear'
    if predicted_value == 5:
        return 'surprise'
    return 'neutral'

TOKEN = '5964787002:AAF-FAIg1IKIGOGpg8zwTaK3LvD-D90HbZ8'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_fullname = message.from_user.full_name
    await message.reply(f"Hello, {user_fullname}!")

@dp.message_handler()
async def echo(message: types.Message):
   text = message.text
   sequence = tokenizer.texts_to_sequences([text])
   data = pad_sequences(sequence, maxlen = max_messages_len)
   result = loaded_model.predict(data)
   predicted_value = result.argmax()
   await message.answer(show_result(predicted_value))

if __name__ == '__main__':
   executor.start_polling(dp, skip_updates=True)