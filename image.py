import numpy as np
import cv2


def resize_image_in_width(image, width):
    # вычисляем отношение между исходным изображением и конечным
    r = float(width) / image.shape[1]  # высчитываем отношение размеров изображений посредством деления высот

    # создаем tuple из 2ух элементов, второй элемент - это расчет новой высоты изображения
    dim = (width, int(image.shape[0] * r))

    # изменяем размер изображения, 1 - изображени, 2 - tuple, 3 - тип интреполяции изображения
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def find_largest_contour(binary_image):
    # Находим контуры на изображении
    image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры не найдены, то ничего не возвращаем
    if not contours:
        return False, None

    # Если контуры найдены то находим и возвращаем максимальный контур
    return True, max(contours, key=cv2.contourArea)


def get_thresh_led():

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # переводим из BGR пространства в оттенки серого

    kernel = (15, 15)  # создаем ядро (размерность по горизонтали и вертикали)
    # Размываем изображени по Гауссу, чем больше ядро, тем больше размытие
    # третий параметр - стандартное отклонени по оси X
    blurred = cv2.GaussianBlur(gray, kernel, 0)

    # создаем двумерный массив - ядро нужной размерности и заполняем его еденицами
    kernel = np.ones((5, 5), np.uint8)

    # Erode — размывание(операция сужения),
    # изображение формируется из локальных минимумов — т.е. будут увеличиваться тёмные области
    erosion = cv2.erode(blurred, kernel, iterations=1)

    # Dilate — растягивание(операция расширения),
    # изображение формируется из локальных максимумов — т.е. будут увеличиваться светлые области
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Выбираем пиксели (ниже или выше) определённого порогового значения,
    # последний аргумент задает логику работы фильтрации
    return cv2.threshold(dilation, 50, 255, cv2.THRESH_TRIANGLE)[1]


def get_largest_contour_center(thresh_image):
    # находим максимально большой контур
    is_contour_founded, largest_contour = find_largest_contour(thresh_image)

    if not is_contour_founded:
        return False, None

    # Вычислеям моменты контура
    moments = cv2.moments(largest_contour)

    # Вычисляем центр контура
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    return True, (center_x, center_y)


if __name__ == '__main__':

    # Читаем изображение из файла,
    # второй параметр функции задает пространство BGR (можно сразу использовать любое другое)
    image = cv2.imread('./test_images/image.jpg', cv2.IMREAD_COLOR)

    # Изменяем размер изображения, масштабируя его относительно новой высоты
    resized_image = resize_image_in_width(image, 1000)

    # Производим фильтрацию для поиска необходимых пятен
    thresh_led_image = get_thresh_led(resized_image)

    # Находим центр самого большого контура
    is_center_founded, center = get_largest_contour_center(thresh_led_image)

    if is_center_founded:
        cv2.circle(resized_image, center, 3, (0, 255, 0), -1)
        cv2.imshow('image', resized_image)

    # Ожидаем нажатие кнопки для закрытия приложения
    key = cv2.waitKey(0)
    if key == 27:  # ожидаем нажатие ESC для выхода
        cv2.destroyAllWindows()
    elif key == ord('s'):  # нажать  's' для выхода
        cv2.destroyAllWindows()
