import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from datetime import datetime, timedelta


def collect_video_characteristics(video_path):
    # Загружаем видео
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Инициализируем массивы для хранения характеристик шума
    # stds = []
    # variances = []
    # gaussians = []
    # hists = []
    features = []

    # Проходимся по всем кадрам видео
    while cap.isOpened():
        # Считываем кадр
        ret, frame = cap.read()

        # Если кадр считан успешно
        if ret:
            height, width = frame.shape[:2]
            fragment_w = width // 5
            fragment_h = height // 5

            # Преобразуем кадр в оттенки серого
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Разбиение изображения на 25 фрагментов
            fragments = []
            for i in range(5):
                for j in range(5):
                    x = j * fragment_w
                    y = i * fragment_h
                    fragment = gray_frame[y:y + fragment_h, x:x + fragment_w]
                    fragments.append(fragment)

            # Подсчет характеристик на каждом из 25 фрагментов кадра
            # frame_stds = []
            # frame_variances = []
            # frame_gaussians = []
            # frame_hists = []

            frame_stats = []
            for fragment in fragments:
                frame_stats.append(np.var(fragment))
                frame_stats.append(np.std(fragment))
                frame_stats.append(np.var(cv2.GaussianBlur(fragment, (5, 5), 0)))
                hist, _ = np.histogram(fragment.ravel(), 256, [0, 256])
                frame_stats.append(np.var(hist))

                # frame_variances.append(np.var(fragment)) # Дисперсия
                # frame_stds.append(np.std(fragment)) # Стандартное отклонение
                # frame_gaussians.append(np.var(cv2.GaussianBlur(fragment, (5, 5), 0))) # Гауссовское размытие
                # hist, _ = np.histogram(fragment.ravel(), 256, [0, 256]) # Гистограмма
                # frame_hists.append(np.var(hist))

            # stds.append(frame_stds)  # Стандартное отклонение
            # variances.append(frame_variances)  # Дисперсия
            # gaussians.append(frame_gaussians) # Гауссовское размытие
            # hists.append(frame_hists) # Гистограмма
            features.append(frame_stats)

            # Отображаем текущий кадр
            # cv2.imshow('frame', frame)

            # Ждем 1 мс перед следующей итерацией
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Освобождаем захваченные ресурсы
    cap.release()
    cv2.destroyAllWindows()

    return {
        "features": features,
        "fps": fps
    }


def detect_anomaly(video_characteristics):
    video_features = np.array(video_characteristics["features"])

    forest = IsolationForest()
    anomaly_scores = forest.fit_predict(video_features)
    anomaly_frames = [index for index, value in enumerate(anomaly_scores) if value == -1]
    print('Аномальные кадры (Isolation Forest): ', get_timecode(video_characteristics["fps"], anomaly_frames))

    lof = LocalOutlierFactor()
    anomaly_scores = lof.fit_predict(video_features)
    anomaly_frames = [index for index, value in enumerate(anomaly_scores) if value == -1]
    print('Аномальные кадры (Local Outlier Factor): ', get_timecode(video_characteristics["fps"], anomaly_frames))

    clf = OneClassSVM()
    anomaly_scores = clf.fit_predict(video_features)
    anomaly_frames = [index for index, value in enumerate(anomaly_scores) if value == -1]
    print('Аномальные кадры (One Class SVM): ', get_timecode(video_characteristics["fps"], anomaly_frames))


def get_timecode(fps, frame_indexes):
    timecodes = []
    start_index = None
    end_index = None

    for i, index in enumerate(frame_indexes):
        if start_index is None:
            start_index = index
            end_index = index
        elif index == end_index + 1:
            end_index = index
        else:
            timecodes.append(format_timecode(start_index, end_index, fps))
            start_index = index
            end_index = index

        if i == len(frame_indexes) - 1:
            timecodes.append(format_timecode(start_index, end_index, fps))

    return timecodes


def format_timecode(start_index, end_index, fps):
    start_time = start_index / fps
    end_time = (end_index + 1) / fps
    return f"{format_seconds(start_time)}-{format_seconds(end_time)}" if start_index != end_index else f"{format_seconds(start_time)}"


def format_seconds(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == '__main__':
    video_characteristics = collect_video_characteristics('video_with_anomaly.mp4')
    detect_anomaly(video_characteristics)
