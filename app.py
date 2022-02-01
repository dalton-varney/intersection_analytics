import time
import edgeiq
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from collections import deque
import numpy as np

app = Flask(__name__, template_folder='./templates/')

socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(
    app, logger=socketio_logger, engineio_logger=socketio_logger)

SESSION = time.strftime("%d%H%M%S", time.localtime())
video_stream = edgeiq.FileVideoStream("intersection.mp4", play_realtime=True)
obj_detect = edgeiq.ObjectDetection("alwaysai/yolo_v3_416_xavier_nx")
print(obj_detect.labels)
obj_detect.load(engine=edgeiq.Engine.TENSOR_RT)
SAMPLE_RATE = 50
peopleTracked = 0
carTracked = 0


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@socketio.on('connect')
def connect_cv():
    print('[INFO] connected: {}'.format(request.sid))


@socketio.on('disconnect')
def disconnect_cv():
    print('[INFO] disconnected: {}'.format(request.sid))


@socketio.on('close_app')
def close_app():
    print('Stop Signal Received')
    controller.close()


class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler
        (https://github.com/alwaysai/video-streamer) and is modified here.

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.all_frames = deque()
        self.video_frames = deque()
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self

    def run(self):
        # loop detection
        global peopleTracked
        global carTracked
        video_stream.start()
        socketio.sleep(0.01)
        self.fps.start()
        print(obj_detect.labels)
        def personenters(person_id, prediction):
            global peopleTracked
            peopleTracked += 1
        def carenters(person_id, prediction):
            global carTracked
            carTracked +=1
        def exits(person_id, prediction):
            """
            Detects when a new person enters.
            Referenced from https://github.com/alwaysai/snapshot-security-camera
            """

        tracker = edgeiq.CentroidTracker(
            deregister_frames=15, enter_cb=personenters, exit_cb=exits)
        cartracker = edgeiq.CentroidTracker(
            deregister_frames=15, enter_cb=carenters, exit_cb=exits)
        while True:
            try:
                frame = video_stream.read()
                text = [""]
                socketio.sleep(0.01)
                width = frame.shape[0]
                height = frame.shape[1]
                # run CV here
                results = obj_detect.detect_objects(
                    frame, confidence_level=.2, overlap_threshold=0.05)
                predictions = results.predictions
                frame = edgeiq.markup_image(frame, predictions, colors=obj_detect.colors, line_thickness=2, font_size=0.7, show_confidences=False)
                tracked_people = tracker.update(edgeiq.filter_predictions_by_label(predictions, ['person']))
                tracked_cars = cartracker.update(edgeiq.filter_predictions_by_label(predictions, ['car']))
                combined = []
                for (key, object) in tracked_people.items():
                    if object.box.start_x != 0 or object.box.start_y != 0 or object.box.end_x != width or object.box.end_y != height:
                        combined.append(object)
                for (key, object) in tracked_cars.items():
                    if object.box.start_x != 0 or object.box.start_y != 0 or object.box.end_x != width or object.box.end_y != height:
                        combined.append(object)
                
                text = ["Yolo V3 | Tensor RT"]
                text.append(
                        "Frames Per Second: {:1.2f} s".format(1/results.duration))
                text.append(f"Current Occupancy People: {len(tracked_people)}")
                text.append(f"Current Occupancy Cars: {len(tracked_cars)}")
                text.append(" ")
                text.append(f"Total Unique People: {int(peopleTracked)}")
                text.append(f"Total Unique Cars: {int(carTracked)}")

                self.send_data(frame, text)
                socketio.sleep(0.01)
                self.fps.update()

                if self.check_exit():
                    video_stream.stop()
                    controller.close()
            except edgeiq.NoMoreFrames:
                video_stream.start()



    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                    frame, width=720, height=480, keep_scale=True)
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text)
                    })
            socketio.sleep(0.01)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.exit_event.is_set()

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()


class Controller(object):
    def __init__(self):
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('[INFO] Starting server at http://localhost:5000')
        socketio.run(app=app, host='0.0.0.0', port=5000)

    def close(self):
        self.fps.stop()
        print("elapsed time: {:.2f}".format(self.fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(self.fps.compute_fps()))

        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()

        print("Program Ending")


controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        controller.close()
