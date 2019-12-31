import argparse

import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize

import time
import dlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-predictor", required=True, help="shape_predictor_68_face_landmarks.dat")
args = parser.parse_args()

print("starting program.")
print("'s' starts drawing eyes.")
print("'r' to toggle recording image, and 'q' to quit")

vs = VideoStream().start()
time.sleep(1.5)

# эта часть обнаруживает наше лицо
detector = dlib.get_frontal_face_detector()
# эта определяет расположение нашего лица
predictor = dlib.shape_predictor(args.predictor)

recording = False
counter = 0


class EyeList(object):
    def __init__(self, length):
        self.length = length
        self.eyes = []

    def push(self, newcoords):
        if len(self.eyes) < self.length:
            self.eyes.append(newcoords)
        else:
            self.eyes.pop(0)
            self.eyes.append(newcoords)

    def clear(self):
        self.eyes = []


# начинает от 10 предыдущих позиций глаз
eyelist = EyeList(10)
eyeSnake = False

# получаем первый кадр вне цикла, так что мы можем увидеть как
# вебкамера изменила свое расширение, которое теперь составляет w / np.shape
frame = vs.read()
frame = resize(frame, width=800)

eyelayer = np.zeros(frame.shape, dtype='uint8')
eyemask = eyelayer.copy()
eyemask = cv2.cvtColor(eyemask, cv2.COLOR_BGR2GRAY)
translated = np.zeros(frame.shape, dtype='uint8')
translated_mask = eyemask.copy()

while True:
    # получаем кадр из камеры, уменьшаем размер
    frame = vs.read()
    frame = resize(frame, width=800)

    # заливаем наши маски и кадры 0 (черный цвет) в каждом цикле рисования
    eyelayer.fill(0)
    eyemask.fill(0)
    translated.fill(0)
    translated_mask.fill(0)

    # детектор ждет серые изображения
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # для случая запуска цикла eyesnake (нажмите «s» во время работы для активации)
    if eyeSnake:
        for rect in rects:
            # предсказатель это наша модель из 68 точек, которую мы загрузили
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # наша модель dlib возвращает 68 точек, которые формируют лицо.
            # левый глаз расположен между 36й и 42й точками.
            # правый глаз – между 42й и 48й точками.
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            # заливаем маску в форме наших глаз
            cv2.fillPoly(eyemask, [leftEye], 255)
            cv2.fillPoly(eyemask, [rightEye], 255)

            # копируем изображение из кадра в eyelayer при помощи этой маски
            eyelayer = cv2.bitwise_and(frame, frame, mask=eyemask)

            # это мы используем для получения координат х и y для вставки глаз
            x, y, w, h = cv2.boundingRect(eyemask)

            # добавляем это в наш список
            eyelist.push([x, y])

            # наконец, рисуем наши глаза в обратном порядке
            for i in reversed(eyelist.eyes):
                # сначала меняем название eyelayer на eyes
                translated1 = translate(eyelayer, i[0] - x, i[1] - y)
                # далее, переводим присвоенную маску
                translated1_mask = translate(eyemask, i[0] - x, i[1] - y)
                # добавляем ее в существующую переведенную маску eyes (не буквально, так как рискуем перегрузить)
                translated_mask = np.maximum(translated_mask, translated1_mask)
                # вырезаем новую переведенную маску
                translated = cv2.bitwise_and(translated, translated, mask=255 - translated1_mask)
                # вставляем в только-что указанную позицию глаза
                translated += translated1
        # вырезаем переведенную маску еще раз
        frame = cv2.bitwise_and(frame, frame, mask=255 - translated_mask)
        # и вставляем в переведенное изображение глаза
        frame += translated

    # показ текущего кадра, проверяем, нажал ли пользователь на клавишу
    cv2.imshow("eye glitch", frame)
    key = cv2.waitKey(1) & 0xFF

    if recordin
        # создаем папку под названием "image_seq", теперь мы можем создавать гифки
        # в ffmpeg с последовательными изображениями
        cv2.imwrite("image_seq/%05d.png" % counter, frame)
        counter += 1

    if key == ord("q"):
        break

    if key == ord("s"):
        eyeSnake = not eyeSnake
        eyelist.clear()

    if key == ord("r"):
        recording = not recording

cv2.destroyAllWindows()
vs.stop()