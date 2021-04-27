import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_video(video_path, block=False, num_blocks=None, index=None):
    '''
    Read video in blocks or directly in memory, if block mode is selected reads only block by index


    :param video_path: path to the video
    :param block: allow block reading
    :param num_blocks: number of blocks. eg. 10
    :param index: index of the block. eg. 2 for the third block
    :return: np.array of frames as uint8 type
    '''
    print('Reading video: ', video_path)
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('video props: (frameCount, frameHeight, frameWidth)=', (frameCount, frameHeight, frameWidth))
    fc = 0
    ret = True
    if not block:
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        while (fc < frameCount and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        cap.release()
    else:
        # calculate block indices:
        block_length = frameCount // num_blocks
        a = index * block_length
        b = a + block_length
        if index == num_blocks - 1:
            b += frameCount % num_blocks
        buf = np.empty((b - a, frameHeight, frameWidth, 3), np.dtype('uint8'))
        cnt = 0
        while (fc < frameCount and ret and fc < b):
            ret, frame = cap.read()
            if fc < b and fc >= a:
                buf[cnt] = frame
                cnt += 1
            fc += 1
    return buf


class BackgroundSubtractorTH:
    def __init__(self, init_frame=None, threshold=0.93):
        self.init_frame = init_frame
        self._track_window = None
        self.threshold = threshold

    def apply(self, frame):
        if self.init_frame is not None:
            frame = frame - self.init_frame

        # mask = cv2.inRange(frame, self.threshold * 255.0, 255.0)
        # ---------> in numpy...
        # frame = frame.astype(float)
        # value = 3*[int(self.threshold*255)]
        # frame[frame < value] = 0
        # # frame[frame >= value] = 1
        # frame_r = frame[:,:,0]
        # frame_g = frame[:,:,1]
        # frame_b = frame[:,:,2]
        # frame_r = frame_r*frame_g*frame_b
        #
        # frame = np.stack([frame_r, frame_r, frame_r], axis=-1)
        # cv2.normalize(frame,frame, 0, 255, cv2.NORM_MINMAX)
        #
        # frame[frame < value] = 0
        # frame[frame >= value] = 1
        # cv2.normalize(frame,frame, 0, 255, cv2.NORM_MINMAX)
        #
        # ===> adaptive opencv....
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        value = int(self.threshold * 255)
        ret, th1 = cv2.threshold(frame_gray, value, 255, cv2.THRESH_BINARY)

        frame = np.stack([th1, th1, th1], axis=-1)
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        # print(frame.max())

        # if self._track_window == None:
        #     x = np.where(frame == np.amax(frame))
        #     self._track_window =  (int(x[1][0])-50, int(x[0][0])-50, 200, 200)
        #     print('track window = ', self._track_window)
        #
        # x, y, w, h = self._track_window
        # track_window = (x, y, w, h)
        #
        # # set up the ROI for tracking
        # roi = frame[y:y + h, x:x + w]
        # hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_roi, np.array((0., 0., 200.)), np.array((30., 50., 255.)))
        # roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        # cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        #
        # # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        # term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        #
        # # # apply meanshift to get the new location
        # # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # # # Draw it on image
        # # x, y, w, h = track_window
        # # img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        #
        #
        # # apply camshift to get the new location
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # # Draw it on image
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        # # img2 = cv2.polylines(frame, [pts], True, 255, 2)
        # x, y, w, h = track_window
        # img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        #
        #
        # self._track_window = track_window
        return frame


def createBackgroundSubtractorTH(init_image=None, bkg_threshold=0.93):
    return BackgroundSubtractorTH(init_frame=init_image, threshold=bkg_threshold)


class MouseVideo:
    def __init__(self, vpath, bkg_method='MOG', bkg_threshold=0.93, roi_dims=(260, 260)):
        self.vpath = vpath
        self.frames = read_video(vpath)
        self.num_frames = self.frames.shape[0]
        self._frames_no_bkg = None
        self._bkg_method = bkg_method
        self.bkg_threshold = bkg_threshold
        self.roi_dims = roi_dims

    def remove_background(self):
        if self._frames_no_bkg is None:
            self._frames_no_bkg = np.empty_like(self.frames)
            if self._bkg_method == 'MOG':
                bg_substractor = cv2.createBackgroundSubtractorMOG2()
                for i in range(self.num_frames):
                    no_bkg_frame = np.tile(bg_substractor.apply(self.frames[i]), (3, 1, 1)).T
                    self._frames_no_bkg[i] = no_bkg_frame
            else:
                bg_substractor = createBackgroundSubtractorTH(bkg_threshold=self.bkg_threshold)
                inverted_frames = 255 - self.frames
                for i in range(self.num_frames):
                    no_bkg_frame = bg_substractor.apply(inverted_frames[i])
                    self._frames_no_bkg[i] = no_bkg_frame

        return self._frames_no_bkg

    def detect_mouse(self, frame_index):
        '''
        Calculate bounding box containing the mouse location.

        :param frame_index: index of the frame in the video
        :return: list of four values formed by bottom left corner and top right corner cords
        '''
        assert frame_index < self.num_frames, f' {frame_index} < {self.num_frames}'
        no_background_frame = self.frames_no_bkg[frame_index]
        gray_image = cv2.cvtColor(no_background_frame, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray_image, 127, 255, 0)

        centroid = cv2.moments(th1)
        # calculate x,y coordinate of center
        try:
            cX = int(centroid["m10"] / centroid["m00"])
            cY = int(centroid["m01"] / centroid["m00"])
        except Exception as e:
            print(centroid)
            plt.imshow(no_background_frame)
            plt.show()
            raise e

        # put text and highlight the center
        frame = self.frames[frame_index]
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(frame, "ROI", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        shift_x, shift_y = (self.roi_dims[0]//2, self.roi_dims[1]//2)
        down_left_x = 0 if cX - shift_x < 0 else cX - shift_x
        down_left_y = 0 if cY - shift_y < 0 else cY - shift_y
        up_right_x = frame.shape[0] if cX + shift_x < 0 else cX + shift_x
        up_right_y = frame.shape[1] if cY + shift_y < 0 else cY + shift_y
        cv2.rectangle(frame, (down_left_x, down_left_y), (up_right_x, up_right_y), 255, 2)
        return frame

    @property
    def frames_no_bkg(self):
        return self._frames_no_bkg

    @frames_no_bkg.getter
    def frames_no_bkg(self):
        self._frames_no_bkg = self.remove_background()
        return self._frames_no_bkg

    @frames_no_bkg.setter
    def frames_no_bkg(self, value):
        assert value is None, 'Set none to recalculate'
        self._frames_no_bkg = None
        self._frames_no_bkg = self.remove_background()

    def invert(self):
        for i in range(self.num_frames):
            self.frames[i] = 255 - self.frames[i]
