from flask import Flask, request, jsonify
import base64
import cv2
from PIL import Image
import numpy as np
import re
from datetime import datetime
import time
from imageio import imread
import io
import keyboard as keyboard
import mediapipe as mp
import os

app = Flask(__name__)

queue = []

@app.route("/", methods=['POST'])
def receive():

    filename = f'result{time.time()}'  # I assume you have a way of picking unique filenames
    lang_code = 1 if (request.get_json(force=True)["lang"] == "eng") else 2
    with open(filename+'.txt', 'w') as f:
        f.write(request.get_json(force=True)["img"])
    with open(filename+'.txt', 'r') as f:
        data_modified = re.sub('^data:image/.+;base64,', '', f.read())
        imgdata = base64.b64decode(data_modified)
        with open(filename+'.jpg', 'wb') as f2:
            f2.write(imgdata)
            queue.append((filename+'.jpg', lang_code))
    os.remove(filename+'.txt')
           
        

    # data_modified = re.sub('^data:image/.+;base64,', '', str(request.data))
    # res = re.sub('^data:image/.+;base64,', '', request.data.decode("utf-8"))
    # pil = Image.open(res)
    # cvImage = cv2.cvtColor(np.array(pil), cv2.COLOR_BGR2RGB)
    # queue.append(cvImage)
        
    return "<p>Hello, World!</p>"

@app.route("/", methods=['GET'])
def send():

    filename, lang_code = queue.pop()
    print("queoe decond eleme is " + str(lang_code))
    img_encode = ""
    res = trans(filename, lang_code)
    with open(filename, "rb") as f:
        img_encode = base64.b64encode(f.read()).decode('utf-8')
    os.remove(filename)

    return jsonify({'msg': 'success', 'char': res, 'size': [1280, 720], 'img': img_encode})

    # return jsonify({})


static_image_mode = True
max_num_hands = 2
english_gesture = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:"G", 7:'H', 8:'I', 9:'J', 10: 'K',
    11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
    21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:' ', 27:'.', 28:'',
}

chinese_gesture = {
    0:'i', 1:'c', 2:'h', 3:'a', 4:'k', 5:'ch', 6:'ng'
}

output_text = ""
delay_time_passed = True
t_end = time.time() + 1
subtitle = ""


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
#     file = np.genfromtxt('data/english_gesture_train.csv', delimiter=',')
# file = np.genfromtxt('data/english_gesture_train.csv', delimiter=',')

# Gesture recognition model


def trans(filePath, lang):
    return_res = ''
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
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # collect_data_file = open('data/collectedData.txt', 'a')
            # if keyboard.is_pressed('s'):
            #     for num in angle:
            #         num = round(num, 6)
            #         collect_data_file.write(str(num))
            #         collect_data_file.flush()
            #         collect_data_file.write(",")
            #         collect_data_file.flush()
            #     collect_data_file.write("6.000000")
            #     collect_data_file.flush()
            #     collect_data_file.write("\n")
            #     collect_data_file.flush()
            #     print("next")
            #     collect_data_file.close()

            # Inference gesture
            # output_text = ""
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # if delay_time_passed:
            #     t_end = time.time() + 1
            #     start_idx = idx

            # Draw gesture result
            # if idx in rps_gesture.keys():
                #cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Other gestures
            output_text = ""
            if idx in english_gesture.keys():
                if time.time() < t_end:
                    delay_time_passed = False
                    # print ("The time now is: " + str(time.time()))
                    # print ("Time to end is" + str(t_end))
                    # unchanged = idx != prev_idx
                    # prev_idx = idx
                else:
                    delay_time_passed = True
                    print("delay time is passed")
                    # if start_idx == idx or len(output_text) == 0 or len(output_text) == 1 :
                    #     if lang == "1":
                    #         output_text += english_gesture[idx].upper()
                    #     else:
                    #         output_text += chinese_gesture[idx].upper()

                        # print(output_text)
                        # subtitle += english_gesture[idx].upper()

                    if len(output_text) > 1:
                        if output_text[-1] == output_text[-2]:
                            output_text = output_text[:-2]

            if lang == 2:
                cv2.putText(img, text=chinese_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]),
                            int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.5, color=(0, 0, 255), thickness=2)
                return_res = chinese_gesture[idx]
            else:
                cv2.putText(img, text=english_gesture[idx].upper(),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255, 255, 255), thickness=2)
                return_res = english_gesture[idx]

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            # img.save("data/result.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filePath, img)

    return return_res

