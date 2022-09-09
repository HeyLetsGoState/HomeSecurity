import datetime
import imutils
import numpy as np
import cv2
import background
import threading
from time import sleep
​
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
from tqdm import tqdm
​
class MotionDetection:
​
    def __init__(self):
        np.random.seed(42)
        self.fps = None
        self.vs = None
        self.firstFrame = None
        self.firstFrame = None
        self.avg = None
        self.zoom = False
        self.isFirst = True
        self.current_frame = None
        self.frames_to_rewind = None
        # And set our base zoom
        self.zoom_level = 1.25
        # we'll make this our buffer, so we can go backward if needed
        self.last_ten_minutes = []
        # 10 minutes of backup 60 seconds & frame rate & 10 minutes
        self.max_frames = None
        self.frames_to_rewind = -1
        self.first_frames = []
        self.occurences = 0
        self.frames_of_interest = 0
​
        # The number of interesting frames in a row
        self.frames_of_data = 0
        self.rewind = False
​
    def check_keys(self, key):
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            return False
        if key == ord(","):
            print('rewinding 1 second')
            self.frames_to_rewind -= self.fps
            try:
                self.current_frame = self.last_ten_minutes[int(self.frames_to_rewind)]
            except:
                self.frames_to_rewind = len(self.last_ten_minutes) + 1
                self.current_frame = self.last_ten_minutes[-self.frames_to_rewind + 1]
        if key == ord("r"):
            print("rewinding 10 seconds")
            self.frames_to_rewind -= (self.fps * 10)
            print(f'Number of frames behind {self.frames_to_rewind}')
            try:
                self.current_frame = self.last_ten_minutes[int(self.frames_to_rewind)]
            except:
                # We can only go back to the start (we rewound too far)
                self.frames_to_rewind = len(self.last_ten_minutes) + 1
                self.current_frame = self.last_ten_minutes[-self.frames_to_rewind + 1]
        if key == ord("f") and self.frames_to_rewind < -1:
            print("fast-forward 10 seconds")
            self.frames_to_rewind += (self.fps * 10)
            self.frames_to_rewind = abs(self.frames_to_rewind) * -1
            print(f'Number of frames behind {self.frames_to_rewind}')
            try:
                self.current_frame = self.last_ten_minutes[int(self.frames_to_rewind)]
            except Exception as e:
                self.frames_to_rewind = -1
                self.current_frame = self.last_ten_minutes[-2]
        if key == ord('.'):
            # jump to current
            self.frames_to_rewind = -1
            self.current_frame = self.last_ten_minutes[-2]
        if key == ord("s"):
            try:
                print(f'Manually saving the last {len(self.last_ten_minutes)} of frames')
                self.record_video(self.last_ten_minutes)
            except Exception as e:
                print(f'timeout due to {e}')
        if key == ord("z"):
            # flip the bit
            self.zoom = not self.zoom
        if key == ord(">") and self.zoom:
            self.zoom_level += .15
        if key == ord("<") and self.zoom:
            if self.zoom_level > 1.25:
                self.zoom_level -= .15
​
    def configure_server(self):
        self.vs = cv2.VideoCapture("rtsp://192.168.86.80/live0")
        self.fps = int(self.vs.get(cv2.CAP_PROP_FPS))
        self.max_frames = 60 * self.fps * 10
​
    def start_the_camera(self):
​
        while True:
            try:
                # Always a rolling 10 minute rewind
                if len(self.last_ten_minutes) >= self.max_frames:
                    # and when full just toss it to the memory card
                    try:
                        print(f'dumping the last {len(self.last_ten_minutes)} of frames')
                        self.record_video(self.last_ten_minutes)
                    except:
                        print('timeout')
​
                # Keep grabbing the most recent frame in the video
                try:
                    frame = self.vs.read()
                    if frame[1] is None:
                        self.vs.release()
                        self.configure_server()
                except Exception as ex:
                    print(f'This is bad {ex}')
​
                w = None
                h = None
                # The current frame and it's an array, so we need the 2nd element in it
                frame = frame[1]
​
                try:
                    frame = imutils.resize(frame, width=1280)
                    w, h = frame[1].shape[:2]
                except:
                    continue
                # If we're done we're done
                if frame is None:
                    break
​
                # First thing I want is to capture the first 30 frames (~2 seconds) for a baseline
                if len(self.first_frames) < 45:
                    self.first_frames.append(frame)
​
                # Then let me calculate the median, so I can remove outliers
​
                if len(self.first_frames) == 45:
                    if self.isFirst:
                        medianFrame = np.median(self.first_frames, axis=0).astype(dtype=np.uint8)
                        # Then make it black and white because it's easier to detect
                        grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
​
                    # So I want to take my current frame and make that gray too
                    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Calculate absolute difference of current frame and the median frame
                    dframe = cv2.absdiff(gframe, grayMedianFrame)
                    # And apply a gaussian to smooth it out and remove janky edges
                    blurred = cv2.GaussianBlur(dframe, (25, 25), 0)
                    # Then get the threshhold from the blurred image
                    ret, tframe = cv2.threshold(blurred, 27, 255, cv2.THRESH_BINARY)
                    # Identifying contours from the threshold
                    (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # For each contour draw the bounding box
                    # loop over the contours ONLY if we have a continuous change (drop the noise)
​
                    # SO I want to make sure I have at least 5 frames going on to make it valuable
                    # If not then screw it it's probably just noise
                    if len(cnts) > 0:
                        self.frames_of_interest +=1
                        if self.frames_of_interest == 5:
                            self.occurences += 1
                    else:
                        self.frames_of_interest = 0
                    for c in cnts:
                        # compute the bounding box for the contour, draw it on the frame,
                        # and update the text
                        (x, y, w, h) = cv2.boundingRect(c)
                        # Disregard items in the top of the picture and small blips
                        # I don't care about the top 125 pixels and the very far right slice
                        if y > 160 and (w * h > 600) and w > 20 and h > 20:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
​
                    self.isFirst = False
​
                    self.last_ten_minutes.append(frame)
​
                    if len(self.last_ten_minutes) % 45 == 0:
                        self.isFirst = True
​
                    key = cv2.waitKey(1) & 0xFF
                    try:
                        response = self.check_keys(key)
                        if response is False:
                            break
                    except:
                        None
​
                    try:
                        self.current_frame = self.last_ten_minutes[int(self.frames_to_rewind)]
                    except:
                        self.frames_to_rewind = int(len(self.last_ten_minutes) * -1)
                        try:
                            self.current_frame = self.last_ten_minutes[self.frames_to_rewind -1]
                        except:
                            # we done goofed - just make it the most recenth
                            self.current_frame = self.last_ten_minutes[-1]
​
                    cv2.imshow("Security Feed",
                                self.zoom_at(self.current_frame, zoom=self.zoom_level) if self.zoom else self.current_frame)
                    self.first_frames.pop(0)
​
                rewind = False
                # cleanup the camera and close any open windows
            except Exception as ex:
                print(f'Error :: {ex}  so just make our most recent frame the frame')
                self.current_frame = frame
        self.vs.release()
        cv2.destroyAllWindows()
​
    def record_video(self, frames):
        print(f'inside record video for {len(frames)} we have {self.occurences} occurences')
        if self.occurences >= 1:
            print(f' we have data to record')
            w, h = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '_recording.avi'
            out = cv2.VideoWriter(filename, fourcc, float(self.fps), (1280, 1024))
​
            for frame in tqdm(frames):
                vidout = cv2.resize(frame, (1280, 1024))  # create vidout funct. with res=300x300
                out.write(vidout)  # write frames of vidout function
​
            # We filled up the video
            out.release()
            # Don't need it anymore
            self.last_ten_minutes.clear()
            # Just make it the most recent frame
            self.last_ten_minutes.append(frame)
​
    def zoom_at(self, img, zoom=1, angle=0, coord=None):
        cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
​
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
​
    def start_thread(self, func, name=None, args=[]):
        print(f'Going to start the thread')
        threading.Thread(target=func, name=name, args=args).start()
        print('done')
​
​
if __name__ == '__main__':
    motion_detection = MotionDetection()
    motion_detection.configure_server()
    motion_detection.start_the_camera()
