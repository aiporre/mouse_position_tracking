
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def write_video(filepath, shape, fps=30):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_height, video_width, CHANNELS = shape
    video_filename = filepath
    writer = cv2.VideoWriter(video_filename, fourcc, fps, (video_width, video_height), isColor=True)
    return writer


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
        while fc < frameCount and ret:
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
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        value = int(self.threshold * 255)
        ret, th1 = cv2.threshold(frame_gray, value, 255, cv2.THRESH_BINARY)

        frame = np.stack([th1, th1, th1], axis=-1)
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        return frame


def createBackgroundSubtractorTH(init_image=None, bkg_threshold=0.93):
    return BackgroundSubtractorTH(init_frame=init_image, threshold=bkg_threshold)


def _remove_blackchannel(img):
    w = 0.98
    miu = 0.95
    ##
    # Calculate A
    I_inv = 255.0 - img
    I_min = np.min(I_inv, axis=2)

    kernel = np.ones((3, 3), np.float32) / 9

    def medfilt2(_img):
        return cv2.filter2D(_img, -1, kernel)

    # I_min_med=medfilt2(I_min);
    I_min_med = medfilt2(I_min)

    A = np.max(I_min_med)

    # %%
    # %Calculate Ac
    I_inv_r = I_inv[:, :, 2]  # takes red channel
    # I_r_med=medfilt2(I_inv_r) # applies the median filter to that
    I_r_med = cv2.filter2D(I_inv_r, -1, kernel)
    A_r = np.max(I_r_med)

    I_inv_g = I_inv[:, :, 1]
    I_g_med = medfilt2(I_inv_g)
    A_g = np.max(I_g_med)

    I_inv_b = I_inv[:, :, 0];
    I_b_med = cv2.filter2D(I_inv_b, -1, kernel)
    A_b = np.max(I_b_med)

    I_inv_A = np.empty_like(img, dtype='float32')
    I_inv_A[:, :, 2] = I_inv_b / A_r
    I_inv_A[:, :, 1] = I_inv_b / A_g
    I_inv_A[:, :, 0] = I_inv_b / A_b

    ##
    I_dark_til = np.min(I_inv_A, axis=2)
    I_med = medfilt2(I_dark_til)

    I_detail = medfilt2(np.abs(I_med - I_dark_til))
    I_smooth = I_med - I_detail

    I_dark_cal = np.empty((img.shape[0], img.shape[1], 2))
    I_dark_cal[:, :, 0] = miu * I_dark_til
    I_dark_cal[:, :, 1] = I_smooth

    I_dark = np.min(I_dark_cal, 2);
    t = 1.0 - w * I_dark

    J_inv = (I_inv - A) / np.stack([t, t, t], axis=-1) + A

    J = 255.0 - J_inv

    return np.clip(J, 0, 255)
    # return J.astype('uint8')



class MouseVideo:
    def __init__(self, vpath, bkg_method='MOG', bkg_threshold=0.93, roi_dims=(260, 260)):
        assert os.path.exists(vpath), "Input video path is non-existing or bad argument {}".format(vpath)
        self.vpath = vpath
        self.frames = read_video(vpath)
        self.num_frames = self.frames.shape[0]
        self._frames_no_bkg = None
        self._bkg_method = bkg_method
        self.bkg_threshold = bkg_threshold
        self.roi_dims = roi_dims

    def remove_darkchannel(self, inplace = False):
        darkframes = np.empty_like(self.frames)
        for i, f in enumerate(self.frames):
            darkframes[i] = _remove_blackchannel(f)
        if inplace:
            self.frames = darkframes
        return darkframes

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

    def detect_mouse(self, frame_index, plot=False):
        """
        Calculate bounding box containing the mouse location.

        :param plot: activate calculation of frame and text on the out put image
        :param frame_index: index of the frame in the video
        :return: list of four values formed by bottom left corner and top right corner cords
        """
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
        shift_x, shift_y = (self.roi_dims[0] // 2, self.roi_dims[1] // 2)
        down_left_x = 0 if cX - shift_x < 0 else cX - shift_x
        down_left_y = 0 if cY - shift_y < 0 else cY - shift_y
        up_right_x = frame.shape[0] if cX + shift_x >= frame.shape[0] else cX + shift_x
        up_right_y = frame.shape[1] if cY + shift_y >= frame.shape[1] else cY + shift_y
        roi_cords = (down_left_x, down_left_y), (up_right_x, up_right_y)
        if plot:
            cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(frame, "ROI", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            return cv2.rectangle(frame, (down_left_x, down_left_y), (up_right_x, up_right_y), 255, 2), roi_cords
        else:
            return roi_cords

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
        '''
        invert frames stored
        :return:
        '''
        for i in range(self.num_frames):
            self.frames[i] = 255 - self.frames[i]

    def save(self, filepath, fps=30):
        writer = write_video(filepath, self.frames[0].shape, fps=fps)
        for frame in self.frames:
            writer.write(frame.astype('uint8'))
        writer.release()

