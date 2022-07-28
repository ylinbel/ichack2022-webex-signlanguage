import cv2
import mediapipe as mp
import numpy as np

static_image_mode = True
max_num_hands = 1

english_gesture = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: "G", 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: '.', 28: '',
}

chinese_gesture = {
    0: 'i', 1: 'c', 2: 'h', 3: 'a', 4: 'k', 5: 'ch', 6: 'ng'
}

output_text = ""
# sub = ""
# subtitle = ""

# Function to store the chars
time_for_same_letter = 0
sub_frame_count = 0
# sentence = []
log = []


# def store_char(ch):
#     global time_for_same_letter
#     global sub
#
#     if len(log) != 0:
#         if ch == log[-1]:
#             time_for_same_letter += 1
#             if time_for_same_letter >= 60:
#                 time_for_same_letter = 0
#                 log.clear()
#                 sentence.append(ch)
#                 sub += ch
#
#         else:
#             log.append(ch)
#             log.clear()
#
#     if sentence[-1] == '.' and (len(sentence) != 0):
#         sub = ""
#         sentence.clear()
#
#     return sub

def store_char(ch):
    global time_for_same_letter
    global sub

    if len(log) != 0:
        if ch == log[-1]:
            time_for_same_letter += 1
            if time_for_same_letter >= 60:
                time_for_same_letter = 0
                return ch

        else:
            log.clear()
            log.append(ch)
    else:
        log.append(ch)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Decide on the language
# lang = input("Please select the language!\n"
#              "1 - English\n"
#              "2 - Chinese\n"
#              "(If not selected, defaulted to English")
#
#
# if lang == "2":
#     print("chinese selected in train data")
#     file = np.genfromtxt('data/chinese_gesture_train.csv', delimiter=',')
# else:
# #     file = np.genfromtxt('data/english_gesture_train.csv', delimiter=',')
# lang = "1"
# file = np.genfromtxt('data/english_gesture_train.csv', delimiter=',')

# # Gesture recognition model
# angle = file[:, :-1].astype(np.float32)
# label = file[:, -1].astype(np.float32)
# knn = cv2.ml.KNearest_create()
# knn.train(angle, cv2.ml.ROW_SAMPLE, label)


def trans(filePath, lang):
    global sub_frame_count
    global subtitle

    if lang == 2:
        print("chinese selected in train data")
        file = np.genfromtxt('data/chinese_gesture_train.csv', delimiter=',')
    else:
        file = np.genfromtxt('data/english_gesture_train.csv', delimiter=',')

    angle = file[:,:-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    img = open(filePath, 'a')
    img = cv2.imread(filePath)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Making sentence
            # output_text = ""
            # if len(output_text) > 1:
            #     if output_text[-1] == output_text[-2]:
            #         output_text = output_text[:-2]


            # Output gesture
            if lang == "2":
                cv2.putText(img, text=chinese_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]),
                                                                         int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=3.5, color=(0, 0, 255), thickness=2)
                sub = store_char(chinese_gesture[idx])
            else:
                cv2.putText(img, text=english_gesture[idx].upper(),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.5, color=(255, 255, 255), thickness=2)
                sub = store_char(english_gesture[idx])

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)




    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filePath, img)

# if __name__ == '__main__':
#     trans(filePath, 1)
