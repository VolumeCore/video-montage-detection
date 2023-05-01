import numpy as np
import cv2


def check_if_video_edited_by_noise_and_gaussian(video_path):
    # Загружаем видео
    cap = cv2.VideoCapture(video_path)

    # Инициализируем массивы для хранения характеристик шума
    stds = []
    variances = []
    gaussian = []

    # Проходимся по всем кадрам видео
    while cap.isOpened():
        # Считываем кадр
        ret, frame = cap.read()

        # Если кадр считан успешно
        if ret:
            # Преобразуем кадр в оттенки серого
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            stds.append(np.std(gray_frame))  # стандартное отклонение
            variances.append(np.var(gray_frame))  # дисперсия
            gaussian.append(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


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

    max_std = np.max(stds)
    min_std = np.min(stds)
    max_var = np.max(variances)
    min_var = np.min(variances)
    max_gaussian = np.max(gaussian)
    min_gaussian = np.min(gaussian)

    print('Максимальное стандартное отклонение шума: ', max_std)
    print('Минимальное стандартное отклонение шума: ', min_std)
    print('Максимальная дисперсия шума: ', max_var)
    print('Минимальная дисперсия шума: ', min_var)
    print('Максимальный уровень Гауссовского размытия: ', max_gaussian)
    print('Минимальный уровень Гауссовского размытия: ', min_gaussian)

    if max_std - min_std > 20 or max_gaussian - min_gaussian > 50:
        return True
    else:
        return False


if __name__ == '__main__':
    if check_if_video_edited_by_noise_and_gaussian('video_edited.mp4'):
        print('Video edited')
    else:
        print('Video is not edited')