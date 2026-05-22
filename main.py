import cv2
import mediapipe as mp
import pygame
import time

# ---------- INIT AUDIO ----------
pygame.mixer.init()
pygame.mixer.set_num_channels(10)

sounds = {
    "do": pygame.mixer.Sound("sounds/do.wav"),
    "re": pygame.mixer.Sound("sounds/re.wav"),
    "mi": pygame.mixer.Sound("sounds/mi.wav"),
    "fa": pygame.mixer.Sound("sounds/fa.wav"),
    "sol": pygame.mixer.Sound("sounds/sol.wav"),
    "la": pygame.mixer.Sound("sounds/la.wav"),
    "si": pygame.mixer.Sound("sounds/si.wav")
}

notes = list(sounds.keys())

# ---------- INIT MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

# tracking mouvement
prev_y = {}
last_played = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):

            label = result.multi_handedness[i].classification[0].label

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # init mémoire
            if i not in prev_y:
                prev_y[i] = y
            if i not in last_played:
                last_played[i] = 0

            dy = y - prev_y[i]

            # affichage doigt
            cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)

            # zones (piano)
            zone_width = w // 7
            zone_index = x // zone_width
            zone_index = min(zone_index, 6)

            note = notes[zone_index]

            # ✅ jouer UNIQUEMENT si descente du doigt
            if dy > 15 and time.time() - last_played[i] > 0.3:
                sounds[note].play()
                last_played[i] = time.time()

            prev_y[i] = y

            # affichage main gauche / droite
            cv2.putText(frame, label, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # dessin main
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # dessiner les zones
    for i in range(7):
        cv2.line(frame, (i * (w // 7), 0), (i * (w // 7), h), (255, 255, 255), 1)

    cv2.imshow("Air Piano", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()