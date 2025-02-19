"""
Edge Facial Expression Recognition

Developed by iCAS LAB

Authors: Peyton Chandarana, Mohammadreza Mohammadi
"""
################################################################################
# STD Libs
import time
import os
import platform
import random

# Third Party
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Local
from sendclass import send_fer_class

################################################################################
def init_usbcamera():
    """
    Initializes a USB Camera.
    """
    print('LOG --> In init_usbcamera')
    vid = cv2.VideoCapture(0, cv2.CAP_V4L2)
    vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    vid.set(cv2.CAP_PROP_FPS, 30)
    print(
        f'Resolution: {vid.get(cv2.CAP_PROP_FRAME_WIDTH)}'
        +f'x{vid.get(cv2.CAP_PROP_FRAME_HEIGHT)}'
        +f' {vid.get(cv2.CAP_PROP_FPS)}'
    )
    return vid

def init_emojis(emoji_size=(40, 40)):
    """
    Get the emojis.
    """
    #print('LOG --> In init_emojis')
    emoji_dir = "../data/emojis"
    emojis_dict = {}
    for png in os.listdir(emoji_dir):
        emotion = png.split(".")[0]
        emojis_dict[emotion] = cv2.resize(
            cv2.imread(os.path.join(emoji_dir, png)),
            emoji_size,
            interpolation=cv2.INTER_AREA,
        )
    return emojis_dict


################################################################################
# PREPROCESSING


def preprocess(in_image):
    """
    Preprocesses the images.
    """
    #print('LOG --> In preprocess')
    out_image = in_image.copy()
    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
    return out_image


def get_faces(in_image):
    """
    Get the face in the frame using Haar Cascade Classifier.
    """
    #print('LOG --> In get_faces')
    haar_files = "../data/haar/haarcascade_frontalface_alt.xml"
    haar_face_cascade = cv2.CascadeClassifier(haar_files)
    faces = haar_face_cascade.detectMultiScale(
        in_image, scaleFactor=1.1, minNeighbors=5
    )
    return faces


################################################################################
# DISPLAY CAMERA


def overlay(
    screen_image,
    detect_message,
    detected_class,
    emoji_dict,
    emoji_size,
    emotion_dict,
    resolution=(1024, 752),
):
    """
    Overlays text and images onto the screen.
    """
    #print('LOG --> In overlay')
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, 35)
    overlay_img = screen_image.copy()
    if detected_class in range(0,3):
        emoji = emoji_dict[emotion_dict[detected_class]["emotion"]]
        # Replace the top right corner pixels with the emoji
        overlay_img[
            0 : emoji_size[1],  # Vertical
            resolution[0] - emoji_size[0] : resolution[0],  # Horizaontal
        ] = emoji
    elif detected_class == 3:
        # Party mode
        emoji = emoji_dict["party"]
        overlay_img[
            0 : emoji_size[1],  # Vertical
            resolution[0] - emoji_size[0] : resolution[0],  # Horizaontal
        ] = emoji

    overlay_img = cv2.putText(
        overlay_img,
        detect_message,
        org,
        font,
        1.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
        False,
    )
    return overlay_img


################################################################################
# FER CLASSIFICATION


def classify_faces(faces, screen_image, interpreter, input_index):
    """
    Classify the faces in an input image.
    """
    #print('LOG --> In classify_faces')
    detected_class = 0
    t_delta, t_1, t_2 = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y, w, h in faces:
        cv2.rectangle(screen_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = screen_image[y : y + h, x : x + w]
        # Do some preprocessing before inference
        input_image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        stretch_near = cv2.resize(input_image, (48, 48), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(stretch_near, axis=0)
        # Load data
        interpreter.set_tensor(input_index, input_data)
        # Perform inference
        t_1 = time.perf_counter()
        interpreter.invoke()
        t_2 = time.perf_counter()
        t_delta = round((t_2 - t_1) * 1000, 2)
        output_details = interpreter.get_output_details()[0]
        detected_class = np.argmax(
            np.squeeze(interpreter.tensor(output_details["index"])())
        )
        org = (x + int(w / 2) - 60, y + h + 25)
        screen_image = cv2.putText(
            screen_image,
            f"{t_delta} ms",
            org,
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
            False,
        )
    return (detected_class, t_delta, screen_image)


################################################################################
# Reset and Send Class


def reset_and_send(emotion_dict):
    """
    Reset the counters and get max class and send
    """
    #print('LOG --> In reset_and_send')
    max_detected_class = 0
    for key, val in emotion_dict.items():
        if val['count'] > emotion_dict[max_detected_class]['count']:
            max_detected_class = key
        emotion_dict[key]['count'] = 0
    # Send the class with the maximum inferences
    if max_detected_class == 1 and random.randint(0,100) < 20:
                max_detected_class = 3
    send_fer_class(max_detected_class)
    return max_detected_class, emotion_dict

def reached_max(emotion_dict, max_val):
    """
    Check if one of the emotions has max inferences.
    """
    for k, v in emotion_dict.items():
        if v['count'] >= max_val:
            return True


################################################################################
# DRIVER FUNCTION
################################################################################


def driver(tflite_model_file):
    """
    Perform Classification
    """
    #print('LOG --> In driver')
    prev_press_time = time.time()
    # Stats
    running_latency = 0
    num_inferences = 0
    # Default Classification State
    perform_inference = False
    detect_message = ""
    # Default Emotion
    detected_class = 1
    # Color
    color = 0
    party_count = 0
    PARTY_THRESH = 30
    # Class Counters
    emotion_dict = {
        0: {"count": 0, "emotion": "angry"},  # Angry/Upset
        1: {"count": 0, "emotion": "happy"},  # Happy
        2: {"count": 0, "emotion": "neutral"},  # Neutral
    }
    MAX_EMOTION_COUNT = 1
    # Emojis
    emoji_size = (64, 64)
    emoji_dict = init_emojis(emoji_size=emoji_size)
    # Get the camera type and init
    camera = init_usbcamera()
    # Edge TPU Libraries
    edgetpu_libs = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    # Setup Edge TPU
    exp_delegate = [tflite.load_delegate(edgetpu_libs, {})]
    interpreter = tflite.Interpreter(
        model_path=tflite_model_file, experimental_delegates=exp_delegate
    )
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    
    while True:
        ret, frame = camera.read()
        screen_image = frame
        # Get the faces
        if perform_inference:
            prep_image = preprocess(screen_image)
            faces = get_faces(prep_image)
            detect_message = "Detecting Emotion..."
            # Detect all faces in the image
            detected_class, latency, screen_image = classify_faces(
                faces, screen_image, interpreter, input_index
            )
            running_latency += latency
            # Increment Detected Class
            emotion_dict[detected_class]["count"] += 1
           # num_inferences += 1
            detected_class = -1
        if reached_max(emotion_dict, MAX_EMOTION_COUNT):
            # Reset Counters and send emotion class
            num_inferences = 0
            running_latency = 0
            detected_class, emotion_dict = reset_and_send(emotion_dict)
            perform_inference = False
            detect_message = "Detected Emotion:"
        # Screen Overlay
        screen_image = overlay(
            screen_image,
            detect_message,
            detected_class,
            emoji_dict,
            emoji_size,
            emotion_dict,
        )
        if detected_class == 3:
            if party_count < PARTY_THRESH:
                screen_image[:,:,color] = screen_image[:,:,color] - 128 
                color = color+1 if color < 2 else 0
                time.sleep(.5)
                party_count += 1
                detect_message = "IT'S TIME TO PARTY!"
            else:
                party_count = 0
                detected_class = 1
                detect_message = "Detected Emotion:"


        # Wait for keypress for 1 ms
        keypress = cv2.waitKey(1) & 0xFF
        press_time = time.time()
        # print(f'Prev: {prev_press_time}, Now: {press_time}, Delta: {press_time - prev_press_time}')
        if keypress != 255 and press_time - prev_press_time > 5.0:
            prev_press_time = press_time
            if keypress == ord("q"):
                break
            if keypress == ord(" "):
                # toggle classification
                perform_inference = True
            if keypress == ord("p"):
                detected_class = 3
                send_fer_class(3)
            if keypress == ord("i"):
                send_fer_class(4)
        elif keypress != 255:
            print('Key pressed too soon...')
        # Show screen image
        cv2.imshow("Frame", screen_image)
