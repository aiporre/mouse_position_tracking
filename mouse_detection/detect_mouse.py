
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import tqdm
from scipy import interpolate

from mouse_detection.tracker import EuclideanDistTracker


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

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
            ret, ff = cap.read()
            if ret:
                buf[fc] = ff
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
    # morphology filters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    def __init__(self, vpath, bkg_method='MOG', bkg_threshold=0.93, roi_dims=(260, 260)):
        assert os.path.exists(vpath), "Input video path is non-existing or bad argument {}".format(vpath)
        self.vpath = vpath
        self.frames = read_video(vpath)
        self.num_frames = self.frames.shape[0]
        self._frames_no_bkg = None
        self._bkg_method = bkg_method
        self.bkg_threshold = bkg_threshold
        self.roi_dims = roi_dims
        self.tracker = EuclideanDistTracker()


    def remove_darkchannel(self, inplace = False):
        darkframes = np.empty_like(self.frames)
        for i, f in tqdm.tqdm(enumerate(self.frames),desc="Removing darkchannel"):
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
                    no_bkg_frame = np.tile(bg_substractor.apply(self.frames[i]), (3, 1, 1)).transpose(1, 2, 0)
                    self._frames_no_bkg[i] = no_bkg_frame
            else:
                bg_substractor = createBackgroundSubtractorTH(bkg_threshold=self.bkg_threshold)
                inverted_frames = 255 - self.frames
                for i in range(self.num_frames):
                    no_bkg_frame = bg_substractor.apply(inverted_frames[i])
                    self._frames_no_bkg[i] = no_bkg_frame

        return self._frames_no_bkg

    def track_mouse(self):
        self.coords = []
        for frame_index in range(self.num_frames):
            try:
                xy1, xy2 = self.detect_mouse(frame_index)
                cX, cY = (xy1[0] + xy2[0])//2 , (xy1[1] + xy2[1])//2
            except ValueError as e:
                cX, cY = np.nan, np.nan
                # raise e
            self.coords.append((cX, cY))
        xx = np.array([x[0] for x in self.coords if not np.isnan(x).sum()>0])
        yy = np.array([x[1] for x in self.coords if not np.isnan(x).sum()>0])
        ii = np.array([i for i, x in enumerate(self.coords) if not np.isnan(x).sum()>0])
        fx = interpolate.interp1d(ii, xx, bounds_error=False, fill_value=(xx[0], xx[-1]))
        fy = interpolate.interp1d(ii, yy, bounds_error=False, fill_value=(yy[0], yy[-1]))
        xx = fx(np.arange(0, len(self.coords)))
        yy = fy(np.arange(0, len(self.coords)))
        xx = savitzky_golay(xx, 65, 3)
        yy = savitzky_golay(yy, 65, 3)
        self.coords=[]
        for x, y in zip(xx, yy):
            self.coords.append((x.astype(int),y.astype(int)))
        return self.coords

    def detect_mouse(self, frame_index, plot=False, crop=False):
        """
        Calculate bounding box containing the mouse location.

        :param plot: activate calculation of frame and text on the out put image
        :param frame_index: index of the frame in the video
        :return: list of four values formed by bottom left corner and top right corner cords
        """
        assert frame_index < self.num_frames, f' {frame_index} < {self.num_frames}'
        no_background_frame = self.frames_no_bkg[frame_index]
        if self._bkg_method == "TH":
            gray_image = cv2.cvtColor(no_background_frame, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(gray_image, 127, 255, 0)
            centroid = cv2.moments(th1)
            # calculate x,y coordinate of center
            try:
                cX = int(centroid["m10"] / centroid["m00"])
                cY = int(centroid["m01"] / centroid["m00"])
            except Exception as e:
                raise e
        else:
            mask = no_background_frame[...,0]
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel10)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                frame_detections = []
                if area > 400:
                    x, y, h, w = cv2.boundingRect(cnt)
                    frame_detections.append([x, y, h, w])
                if len(frame_detections) > 0:
                    detections_avg = np.mean(frame_detections, axis=0)
                    detections.append(detections_avg.astype(int).tolist())
            # detection
            boxes_ids = self.tracker.update(detections)
            frame_coords = []
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                frame_coords.append([x, y, w, h])
            if len(frame_coords) > 0:
                x, y ,h , w = np.mean(frame_coords, axis=0)
                cX, cY = int(x + x + w)//2, int(y + y + h)//2
            else:
                raise ValueError('Frame has not detections')
        if plot:
            frame_plot, roi_coords = self.calculate_roi(frame_index, cX, cY, plot=plot, crop=crop)
            return frame_plot, roi_coords
        else:
            roi_coords = self.calculate_roi(frame_index, cX, cY, plot=plot, crop=crop)
            return roi_coords

    def calculate_roi(self, frame_index, cX, cY, plot=False, crop=False):
        # put text and highlight the center
        frame = self.frames[frame_index]
        shift_y, shift_x = (self.roi_dims[0] // 2, self.roi_dims[1] // 2)
        epsx = 1 if self.roi_dims[0] % 2 == 0 else 0 # if odd no shift and need only one if even you need one more pixel
        epsy = 1 if self.roi_dims[1] % 2 == 0 else 0 # same for y
        down_left_x = 0 if cX - shift_x < 0 else cX - shift_x + epsx
        down_left_y = 0 if cY - shift_y < 0 else cY - shift_y + epsy
        up_right_x = frame.shape[1] if cX + shift_x >= frame.shape[1] else cX + shift_x + 1
        up_right_y = frame.shape[0] if cY + shift_y >= frame.shape[0] else cY + shift_y + 1
        roi_cords = (down_left_x, down_left_y), (up_right_x, up_right_y)
        if plot and not crop:
            cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(frame, "ROI", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            return cv2.rectangle(frame, (down_left_x, down_left_y), (up_right_x, up_right_y), 255, 2), roi_cords
        elif plot and crop:
            crop_dims = list(self.roi_dims) + [frame.shape[-1]] if len(frame.shape) == 3 else self.roi_dims
            crop = np.zeros(crop_dims, dtype=frame.dtype)
            # this doesn't depend on the center even or odd as it is on the left side.
            # Doesn't need +1 as B-Cx=Delta is counting abs from 1 i.e. includes +1 implicitly
            down_left_x_roi = 0 if cX - shift_x >= 0 else shift_x - cX - epsx
            down_left_y_roi = 0 if cY - shift_y >= 0 else shift_y - cY - epsy
            # if even then we have one more pixel in the up-right as the center is an even number in the first quadrant
            # that is because we are using int division, which equivalent to use ceil(x/y)
            # gamma = frame.shape[0] - 1 - cX from cx to the end of frame the rest is zeros
            epsx = self.roi_dims[0] % 2  # if odd no shift and need only one if even you need one more pixel
            epsy = self.roi_dims[1] % 2  # same for y
            up_right_x_roi = self.roi_dims[0] if cX + shift_x < frame.shape[1] else shift_x + frame.shape[1] - 1 - cX + epsx
            up_right_y_roi = self.roi_dims[1] if cY + shift_y < frame.shape[0] else shift_y + frame.shape[0] - 1 - cY + epsy
            crop[down_left_y_roi:up_right_y_roi, down_left_x_roi:up_right_x_roi] = \
                frame[down_left_y:up_right_y, down_left_x:up_right_x]
            return crop, roi_cords
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

    def save(self, filepath, fps=30, no_background=False):
        writer = write_video(filepath, self.frames[0].shape, fps=fps)
        if not no_background:
            for frame in self.frames:
                writer.write(frame.astype('uint8'))
        else:
            for frame in self._frames_no_bkg:
                writer.write(frame.astype('uint8'))
        writer.release()

    def save_rois(self, filepath, fps=30):
        coords = self.track_mouse()
        frame = self.frames[0]
        crop_dims = list(self.roi_dims) + [frame.shape[-1]] if len(frame.shape) == 3 else self.roi_dims

        writer = write_video(filepath, crop_dims, fps=fps)
        for index in range(len(coords)):  # range(self.mouse_video.num_frames):
            cX, cY = coords[index]
            frame, roi = self.calculate_roi(index, cX, cY, plot=True, crop=True)
            writer.write(frame.astype('uint8'))
        writer.release()

