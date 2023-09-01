from keras.models import model_from_json
import face_recognition
import cv2
import numpy as np
import pickle
import tensorflow as tf
import time
from db import create_db

def recognize_faces(frame, statevars, confidence_dict, frame_counter, known_face_encodings, known_face_names, frame_regulator, confidence_repeat, threshold, diff_threshold, model_tolerance, margin):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    overlay = np.zeros((frame_height, frame_width, 3))

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]

    frame_counter += 1
    [face_locations, names] = statevars

    if frame_counter == frame_regulator:
        frame_counter = 0

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        names = []
        for face_encoding in face_encodings:
            name = 'Not Authorized'

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=model_tolerance)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            A, B = np.partition(face_distances, 1)[0:2]
            diff = abs(A - B)

            if matches[best_match_index] and face_distances[best_match_index] < threshold and diff > diff_threshold:
                name = known_face_names[best_match_index]

            names.append(name)
        statevars = [face_locations, names]


    for i in range(len(face_locations)):
        newface = True
        recognized = 2

        (y1, x2, y2, x1) = face_locations[i]
        name = names[i]

        x1 *= 2
        x2 *= 2
        y1 *= 2
        y2 *= 2

        xlim1 = x1 - margin
        xlim2 = x2 + margin
        ylim1 = y1 - margin
        ylim2 = y2 + margin

        region_identifier = str(xlim1) + '_' + str(xlim2) + '_' + str(ylim1) + '_' + str(ylim2)

        for key in confidence_dict.keys():
            [rxlim1, rxlim2, rylim1, rylim2] = key.split('_')
            if x1>int(rxlim1) and x2<int(rxlim2) and y1 > int(rylim1) and y2<int(rylim2):
                region_identifier = key
                newface=False
                break

        if region_identifier not in confidence_dict.keys() and newface:
            confidence_dict[region_identifier] = ['previous', 0, time.time()]

        [previous, confidence_counter, _] = confidence_dict[region_identifier]

        current = name
        if current == previous:
            if confidence_counter < confidence_repeat:
                confidence_counter += 1
            if confidence_counter == confidence_repeat:
                recognized = 1
                if current == 'Not Authorized':
                    recognized = 0
        else:
            confidence_counter = 0
        previous = current

        confidence_dict[region_identifier] = [previous, confidence_counter, time.time()]

        if confidence_counter == confidence_repeat and recognized==1:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (1, 255, 1), 2)
            cv2.rectangle(overlay, (x1, y2 + 40), (x2, y2), (1, 255, 1), cv2.FILLED)
            cv2.putText(overlay, current, (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (1, 1, 1), 1)

        elif confidence_counter == confidence_repeat and recognized==0:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (1, 1, 255), 2)
            cv2.rectangle(overlay, (x1, y2 + 40), (x2, y2), (1, 1, 255), cv2.FILLED)
            cv2.putText(overlay, "Not Authorized", (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (1, 1, 1), 1)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (1, 255, 255), 2)
            cv2.rectangle(overlay, (x1, y2 + 40), (x2, y2), (1, 255, 255), cv2.FILLED)
            cv2.putText(overlay, "Processing...", (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (1, 1, 1), 1)

    return overlay, statevars, confidence_dict, frame_counter


def combine_image_and_overlay(frame, overlay):
    idx = np.nonzero(overlay)
    frame[idx]=overlay[idx]
    return frame

def handle_objects_drawing(overlay, boxes, classes, scores, threshold=0.5):
    img_height = overlay.shape[0]
    img_width = overlay.shape[1]
    for i in range(len(boxes)):
        if scores[i]<threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        cls = classes[i]
        if cls == 1:
            cv2.rectangle(overlay, (int(xmin*img_width), int(ymin*img_height)), (int(xmax*img_width), int(ymax*img_height)), (180, 105, 255), 2)
            cv2.rectangle(overlay, (int(xmin*img_width), int(ymin*img_height) + 20), (int(xmax*img_width)-50, int(ymin*img_height)), (180, 105, 255), cv2.FILLED)
            cv2.putText(overlay, "Helmet", (int(xmin*img_width) + 6, int(ymin*img_height) + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (1, 1, 1), 1)
        elif cls == 2:
            cv2.rectangle(overlay, (int(xmin * img_width), int(ymin * img_height)), (int(xmax * img_width), int(ymax * img_height)), (180, 105, 255), 2)
            cv2.rectangle(overlay, (int(xmin * img_width), int(ymin * img_height) + 20), (int(xmax * img_width)-50, int(ymin * img_height)), (180, 105, 255), cv2.FILLED)
            cv2.putText(overlay, "Vest", (int(xmin * img_width) + 6, int(ymin * img_height) + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (1, 1, 1), 1)
    return overlay

def detect_objects(frame, sess, graph, threshold = 0.5):

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    overlay = np.zeros((frame_height, frame_width, 3))

    frame_expanded = np.expand_dims(frame, axis=0)
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(
                tensor_name)


    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: frame_expanded})

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    overlay = handle_objects_drawing(overlay, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], threshold)

    return overlay

def kill_unused_regions(confidence_dict, timetodie):
    kill_list = []
    for key in confidence_dict.keys():
        [_, _, timestamp] = confidence_dict[key]
        if time.time()-timestamp>timetodie:
            kill_list.append(key)
    for key in kill_list:
        confidence_dict.pop(key, None)
    return confidence_dict


def get_full_path(file_name, file):
    folder = './model/'
    full_path = folder + file_name + file
    return full_path

def detect_eyes_on_face(x, y, w, h):
    y_top = y + int(15 / 64. * h)
    y_bottom = y + int(35 / 64. * h)
    y_h = y_bottom - y_top
    return  x, y_top, w, y_h

def load_keras_model(file_name):
    f = get_full_path(file_name, '.json')
    json_file = open(f, 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    weights_f = get_full_path(file_name, '.h5')
    loaded_model.load_weights(weights_f)
    return loaded_model

def detect_glasses(frame, model, detector):
    frame_height = frame.shape[0]
    frame_weight = frame.shape[1]
    overlay = np.zeros((frame_height, frame_weight, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        # EYEGLASS DETECTION
        e_x, e_y, e_w, e_h = detect_eyes_on_face(x, y, w, h)
        eyes_data = gray[e_y: e_y + e_h, e_x: e_x + e_w]

        eyes_resized = np.array([cv2.resize(eyes_data, (64, 20))])
        eyes_x = eyes_resized.reshape(eyes_resized.shape[0], eyes_resized.shape[1], eyes_resized.shape[2], 1) / 255.
        prediction = model.predict_classes(eyes_x)[0]

        if prediction == 0:
            out = 'No Glasses'
            cv2.putText(overlay, out, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 255), 2, cv2.LINE_AA)
        else:
            out = 'With Glasses'
            cv2.putText(overlay, out, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 255, 1), 2, cv2.LINE_AA)
    return overlay

def main(alias):
    #url = "http://192.168.1.93:8080/video"
    video_capture = cv2.VideoCapture(0)
    #stream_capture = cv2.VideoCapture(url)


    ## FACE DETECTION ##
    [known_face_encodings, known_face_names] = pickle.load(open('./db/'+alias+'.pickle', 'rb'))

    frame_counter = 0
    confidence_dict = {}
    timetodie = 5

    face_params = {
        'known_face_encodings': known_face_encodings,
        'known_face_names': known_face_names,
        'frame_regulator': 5,
        'confidence_repeat':5,
        'threshold': 0.5,
        'diff_threshold': 0.08,
        'model_tolerance': 0.5,
        'margin':20
    }

    statevars = [[], []]
    ## *********** ##

    #PPE DETECTION PARAMS##
    model_path = "./model/frozen_inference_graph.pb"
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            od_graph_def.ParseFromString(fid.read())
            tf.import_graph_def(od_graph_def, name='')

    object_detection_threshold = 0.5
    #***********##

    #GLASS DETECTION#

    glass_model = load_keras_model('glass_cnn')
    detector = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

    #**************#

    sess = tf.compat.v1.Session(graph=graph)

    while True:
        ret, frame = video_capture.read()
        faces_overlay, statevars, confidence_dict, frame_counter = recognize_faces(frame, statevars, confidence_dict, frame_counter, **face_params)
        confidence_dict = kill_unused_regions(confidence_dict, timetodie)
        objects_overlay = detect_objects(frame, sess, graph, threshold=object_detection_threshold)
        glasses_overlay = detect_glasses(frame, glass_model, detector)

        frame = combine_image_and_overlay(frame, objects_overlay)
        frame = combine_image_and_overlay(frame, faces_overlay)
        frame = combine_image_and_overlay(frame, glasses_overlay)

        #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sess.close()
            break

    video_capture.release()
    cv2.destroyAllWindows()

#create_db('db_ilyas')
main('db_yassawe')
