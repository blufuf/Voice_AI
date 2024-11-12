import os
import librosa
import tensorflow as tf
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from keras.models import load_model


model_path = 'voice_authentication_model_lstmV2.h5'  # Укажите здесь путь к вашей модели
model = load_model(model_path)

user_names = ["Blufuf", "Арсений", "Даня", "Денис", "Дима", "Илья", "Кирилл", "Костя", "Никита", "Никита К"]


def load_and_preprocess_audio(file_path, target_length=8000):
    audio_data, _ = librosa.load(file_path, sr=22050)
    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:target_length]
    return audio_data


def extract_features(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
    return mfccs.T


def authenticate_voice(file_path):
    try:
        audio_data = load_and_preprocess_audio(file_path)
        features = extract_features(audio_data)
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        user_id = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.7:
            result = f"Пользователь {user_names[user_id]} аутентифицирован с уверенностью {confidence * 100:.2f}%"
        else:
            result = "Пользователь не зарегистрирован или уверенность слишком низкая."
    except Exception as e:
        result = f"Ошибка обработки файла: {e}"
    return result


def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        result = authenticate_voice(file_path)
        result_label.config(text=result)

root = tk.Tk()
root.title("Аутентификация голоса")
root.geometry("500x500")
root.resizable(False, False)
tk.Label(root, text= 'Аутентификация человека по голосу', font=("Times New Roman", 14)).place(x=20, y=50)
result_label = Label(root, text="Выберите аудиофайл для проверки", wraplength=400)
result_label.pack(pady=20)

load_button = Button(root, text="Загрузить аудиофайл", command=load_file)
load_button.pack()
tk.Label(win, text='Команда разработчиков: *', font=("Times New Roman", 10)).place(x=10, y=470)

def Error():
    showerror(title="Ошибка", message="Вы не выбрали файл")
root.mainloop()