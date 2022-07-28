import cv2
import time
import keyboard as keyboard
import mediapipe as mp
import numpy as np
import pyttsx3 as pyttsx3


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


max_num_hands = 2
english_gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:"G", 7:'H', 8:'I', 9:'J', 10: 'K',
    11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
    21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:' ', 27:'.', 28:'spiderman!', 29:'',
}

chinese_gesture = {
    0:'i', 1:'c', 2:'h', 3:'a', 4:'k', 5:'ch', 6:'ng'
}

sentence = []
output_text = ""
delay_time_passed = True
t_end = time.time() + 2

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# Decide on the language
lang = input("Please select the language!\n"
             "1 - English\n"
             "2 - Chinese\n"
             "(If not selected, defaulted to English")

if lang == "2":
    print("chinese selected in train data")
    file = np.genfromtxt('data/chinese_gesture_train.csv', delimiter=',')
else:
    file = np.genfromtxt('data/english_gesture_train.csv', delimiter=',')


# Gesture recognition model
collect_data_file = open('data/collectedData.txt', 'a')

angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute vectors between each points (joints)
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Getting angle using dot product folmula
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle)

            collect_data_file = open('data/collectedData.txt', 'a')
            if keyboard.is_pressed('s'):
                for num in angle:
                    num = round(num, 6)
                    collect_data_file.write(str(num))
                    collect_data_file.flush()
                    collect_data_file.write(",")
                    collect_data_file.flush()
                collect_data_file.write("29.000000")
                collect_data_file.flush()
                collect_data_file.write("\n")
                collect_data_file.flush()
                print("next")
                collect_data_file.close()

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            if delay_time_passed:
                t_end = time.time() + 2
                start_idx = idx

            # Draw gesture result
            if idx in english_gesture.keys():
                if time.time() < t_end:
                    delay_time_passed = False
                else:
                    delay_time_passed = True
                    print("delay time is passed")
                    if start_idx == idx or len(output_text) == 0 or len(output_text) == 1 :
                        if lang == "1":
                            if idx == 29:
                                output_text = output_text[:-1]
                            else:
                                output_text += english_gesture[idx].upper()
                        else:
                            output_text += chinese_gesture[idx].upper()

                        print(output_text)

                    # Deleting same characters in a row (should not be a problem if right timing used)
                    if len(output_text) > 1:
                        if output_text[-1] == output_text[-2]:
                            output_text = output_text[:-1]

            if lang == "2":
                cv2.putText(img, text=chinese_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]),
                            int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=3.5, color=(0, 0, 255), thickness=2)
            else:
                cv2.putText(img, text=english_gesture[idx].upper(),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.5, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if output_text != '':
                draw_text(img, output_text,
                          font=cv2.FONT_HERSHEY_PLAIN,
                          pos=(50,400),
                          font_scale=4,
                          font_thickness=2,
                          text_color=(255, 255, 255),
                          text_color_bg=(0, 0, 0)
                          )

    cv2.imshow('Sign language translator', img)
    if cv2.waitKey(1) == ord('q'):
        break

    if keyboard.is_pressed('p'):
        engine = pyttsx3.init()
        engine.setProperty("rate", 100)
        engine.say(output_text)
        engine.runAndWait()

    if keyboard.is_pressed('esc'):
        collect_data_file.close()
        print("The output text is: " + output_text)
        break



