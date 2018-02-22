import numpy as np
import image
import cv2

if __name__ == '__main__':

    # Иницилизируем определенную камеру
    camera_number = 0
    cap = cv2.VideoCapture(camera_number)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Изменяем размер изображения, масштабируя его относительно новой высоты
        resized_image = image.resize_image_in_width(frame, 1000)

        # Производим фильтрацию для поиска необходимых пятен
        thresh_led_image = image.get_thresh_led(resized_image)

        # Находим центр самого большого контура
        is_center_founded, center = image.get_largest_contour_center(thresh_led_image)

        # Если контур найден, то визуализируем его данные на изображении
        if is_center_founded:
            cv2.circle(resized_image, center, 3, (0, 255, 0), -1)

        cv2.imshow('image',resized_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Завершаем работу с камерой
    cap.release()
    # Закрываем все окна
    cv2.destroyAllWindows()
