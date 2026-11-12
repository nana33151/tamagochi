import time
from functools import partial
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'face_landmarker.task'

import math

HAND_MODEL = ROOT / 'hand_landmarker.task'
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
)


TIPS = [4, 8, 12, 16, 20]

def get_hand_angle(landmarks):
    wrist = landmarks[0]
    middle_base = landmarks[9]
    dx = middle_base.x - wrist.x
    dy = middle_base.y - wrist.y
    return math.degrees(math.atan2(dy, dx))

def rotate_point(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    new_x = x * math.cos(angle_rad) + y * math.sin(angle_rad)
    new_y = -x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return new_x, new_y

def get_finger_status(landmarks, hand_label):
    angle = get_hand_angle(landmarks)
    correction = -angle + 90

    def r(lm):
        return rotate_point(lm.x, lm.y, correction)

    if hand_label == "Right":
        palm_facing = landmarks[17].x > landmarks[5].x
    else:
        palm_facing = landmarks[17].x < landmarks[5].x

    fingers = []

    tx4, _ = r(landmarks[4])
    tx3, _ = r(landmarks[3])

    if palm_facing:
        if hand_label == "Right":
            fingers.append(1 if tx4 > tx3 else 0)
        else:
            fingers.append(1 if tx4 < tx3 else 0)
    else:
        if hand_label == "Right":
            fingers.append(1 if tx4 < tx3 else 0)
        else:
            fingers.append(1 if tx4 > tx3 else 0)

    for tip_id in TIPS[1:]:
        _, tip_y = r(landmarks[tip_id])
        _, base_y = r(landmarks[tip_id - 2])
        fingers.append(1 if tip_y > base_y else 0)

    return fingers

def detect_gesture(fingers):
    others = fingers[1:]
    # большой палец часто шумит; учитываем и инверсию Y (tip_y <> base_y)
    if others in ([0, 1, 0, 0], [1, 0, 1, 1]):
        return 'Middle Finger'
    if fingers == [0, 0, 0, 0, 0]:
        return 'Punch'
    if fingers == [1, 1, 1, 1, 1]:
        return 'Open Hand'
    if fingers == [0, 1, 1, 0, 0]:
        return 'Peace'
    if fingers == [1, 0, 0, 0, 0]:
        return 'Like'
    return '?'


state = {
    'shrimp': False,          # Флаг для переключения креветка/кот
    'current_state': 'cat',   # Какой спрайт рисовать прямо сейчас
    'animation_until': 0,     # Время (в сек), до которого крутится анимация эмоции
    'frame_index': 0,         # Какой кадр анимации показывать сейчас
    'last_frame_time': 0,
    'too_close': False,       # Лицо слишком близко к камере
}


def trigger_animation(state_name, duration=1.5):
    if state['current_state'] != state_name:
        state['current_state'] = state_name
        state['frame_index'] = 0  # Сбрасываем анимацию на первый кадр
    state['animation_until'] = time.time() + duration


def result_callback(result: vision.FaceLandmarkerResult,
                    output_image: mp.Image,
                    timestamp_ms: int,
                    state_ref: dict):

    SMILE_THRESHOLD = 0.4
    WINK_CLOSED = 0.5  # глаз реально закрыт; 0.5 ловит обычный перекос
    WINK_OPEN = 0.3    # второй глаз должен быть открыт
    TOO_CLOSE_THRESHOLD = 30.0
    TOO_FAR_THRESHOLD = 45.0  # гистерезис, чтобы состояние не дёргалось на границе

    if result.facial_transformation_matrixes:
        z_distance = abs(result.facial_transformation_matrixes[0][2][3])
        if state_ref['too_close']:
            still_close = z_distance < TOO_FAR_THRESHOLD
        else:
            still_close = z_distance < TOO_CLOSE_THRESHOLD

        if still_close:
            if not state_ref['too_close']:
                state_ref['too_close'] = True
                state_ref['current_state'] = 'to_close'
                state_ref['frame_index'] = 0
            return
        if state_ref['too_close']:
            state_ref['too_close'] = False
            state_ref['animation_until'] = 0
            idle = 'shrimp' if state_ref['shrimp'] else 'cat'
            state_ref['current_state'] = idle
            state_ref['frame_index'] = 0
    elif state_ref['too_close']:
        state_ref['too_close'] = False
        state_ref['animation_until'] = 0

    if state_ref['current_state'] == 'middle' and time.time() < state_ref['animation_until']:
        return

    if time.time() > state_ref['animation_until']:
        idle = 'shrimp' if state_ref['shrimp'] else 'cat'
        if state_ref['current_state'] != idle:
            state_ref['frame_index'] = 0
        state_ref['current_state'] = idle

    if result.face_blendshapes:
        bsh_dict = {b.category_name: b.score for b in result.face_blendshapes[0]}

        smile_left = bsh_dict.get('mouthSmileLeft', 0)
        smile_right = bsh_dict.get('mouthSmileRight', 0)

        eye_blink_left = bsh_dict.get('eyeBlinkLeft', 0)
        eye_blink_right = bsh_dict.get('eyeBlinkRight', 0)
        left_wink = eye_blink_left > WINK_CLOSED and eye_blink_right < WINK_OPEN
        right_wink = eye_blink_right > WINK_CLOSED and eye_blink_left < WINK_OPEN
        rotate = False
        if result.facial_transformation_matrixes:
            R = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
            yaw_rad = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
            rotate = abs(np.degrees(yaw_rad)) > 15
        if smile_left > SMILE_THRESHOLD and smile_right > SMILE_THRESHOLD:
            trigger_animation('smile')
        elif (left_wink or right_wink) and not rotate:
            trigger_animation('melt')


base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    result_callback=partial(result_callback, state_ref=state),
)
detector = vision.FaceLandmarker.create_from_options(options)

def load_frames(prefix, count):
    return [cv2.imread(str(ROOT / f'{prefix}_{i}.png')) for i in range(count)]

middle_frames = [f for f in load_frames('angry', 2) if f is not None]
punch_frames = [f for f in load_frames('fight', 5) if f is not None]
if not middle_frames:
    middle_frames = load_frames('melt', 8)

animations = {
    'cat': load_frames('cat', 1),
    'shrimp': load_frames('shrimp', 1),
    'smile': load_frames('smile', 4),
    'melt': load_frames('melt', 8),
    'to_close': load_frames('shrimp', 5),
    'middle': middle_frames,
    'fight': punch_frames,
}
animations = {
    name: [frame for frame in frames if frame is not None]
    for name, frames in animations.items()
}

FRAME_DELAY = 0.2  # Задержка между кадрами персонажа в секундах (0.2 сек = 5 кадров в секунду)
SPRITE_SIZE = 200

cv2.namedWindow('Pet', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pet', SPRITE_SIZE, SPRITE_SIZE)
# Перемещаем окно в правый нижний угол экрана ноутбука (например, координаты X=1200, Y=700)
cv2.moveWindow('Pet', 1200, 700)
# Убираем рамки окна (делает его похожим на виджет)
cv2.setWindowProperty('Pet', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Pet', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 33
    hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
    if hand_result.hand_landmarks:
        for lm_list, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            hand_label = handedness[0].category_name  # "Left" / "Right"
            fingers = get_finger_status(lm_list, hand_label)
            gesture = detect_gesture(fingers)
            if gesture == 'Punch' and not state['too_close']:
                trigger_animation('fight', duration=1.5)
            elif gesture == 'Middle' and not state['too_close']:
                trigger_animation('middle', duration=1.5)

    detector.detect_async(mp_image, timestamp_ms)
    # --- ЛОГИКА АНИМАЦИИ ---
    current_action = state['current_state']
    frame_list = animations.get(current_action, animations['cat'])

    # Защита: если список кадров пустой или картинка не загрузилась, создаем пустой черный квадрат
    if not frame_list:
        pet_frame = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8)
        cv2.putText(pet_frame, current_action, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # Переключаем индекс кадра по таймеру FRAME_DELAY
        if time.time() - state['last_frame_time'] > FRAME_DELAY:
            last_frame = len(frame_list) - 1
            if state['too_close']:
                # Проигрываем переход в shrimp один раз и замираем на последнем кадре
                if state['frame_index'] < last_frame:
                    state['frame_index'] += 1
            else:
                state['frame_index'] = (state['frame_index'] + 1) % len(frame_list)
            state['last_frame_time'] = time.time()

        # Берем текущий кадр из списка и подгоняем под размер окна
        raw_img = frame_list[min(state['frame_index'], len(frame_list) - 1)]
        pet_frame = cv2.resize(raw_img, (SPRITE_SIZE, SPRITE_SIZE))

    # --- ОТРИСОВКА В ДВУХ ОКНАХ ---
    # 1. Показываем персонажа в его собственном окне в углу экрана
    cv2.imshow('Pet', pet_frame)

    # 2. Показываем саму веб-камеру (для отладки, потом эту строку можно удалить)
    #cv2.imshow('Camera Debug', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        state['shrimp'] = not state['shrimp']

cap.release()
detector.close()
hand_detector.close()
cv2.destroyAllWindows()
