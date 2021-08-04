from unittest import TestCase
from mouse_detection.detect_mouse import MouseVideo
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path


class TestMouseVideo(TestCase):
    def setUp(self) -> None:
        self.test_video_mock_up = 'resources/mouse_short_converted_mac.mov'
        if not os.path.exists(self.test_video_mock_up):
            self.test_video_mock_up = str(Path(os.path.join("..", self.test_video_mock_up)).resolve())
        print('path exists: ', os.path.exists(self.test_video_mock_up))
        self.mouse_video = MouseVideo(self.test_video_mock_up, bkg_method='TH')

    def test_detect_mouse(self):
        # index =96
        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            frame, roi = self.mouse_video.detect_mouse(index, plot=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} and roi{roi}')
            plt.show()

    def test_detect_mouse_and_crop(self):
        # index =96
        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            frame, roi = self.mouse_video.detect_mouse(index, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

    def test_detect_mouse_and_crop_with_HOG(self):
        # index =96
        mouse_video_mog = MouseVideo(self.test_video_mock_up, bkg_method='MOG')

        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            try:
                frame, roi = mouse_video_mog.detect_mouse(index, plot=True, crop=True)
                plt.imshow(frame)
                plt.title(f'this is the frame from the index {index}')
                plt.show()
            except ValueError as e:
                print('EROR TEST MOUSE AND CROP WITH HOG = ', e)
                pass

    def test_get_no_background(self):

        method = self.mouse_video._bkg_method
        self.mouse_video._bkg_method = 'MOG'
        self.mouse_video.frames_no_bkg = None

        index = 10
        frame = self.mouse_video.frames_no_bkg[index]
        plt.imshow(frame)
        plt.title(f'this is the frame from the index {index}')
        plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_get_no_background_TH(self):
        method = self.mouse_video._bkg_method
        self.mouse_video._bkg_method = 'TH'
        self.mouse_video.frames_no_bkg = None

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            frame = self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_remove_darkchannel(self):
        blackchannel_imgs = self.mouse_video.remove_darkchannel()

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            frame = blackchannel_imgs[index]  # self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

    def test_track_mouse(self):
        coords = self.mouse_video.track_mouse()
        print(coords)
        plt.plot(coords)
        plt.show()

    def test_track_mouse_MOG(self):
        mouse_video_mog = MouseVideo(self.test_video_mock_up, bkg_method='MOG')
        coords = mouse_video_mog.track_mouse()
        print(coords)
        plt.plot(coords)
        plt.show()

    def test_track_mouse_MOG_plotting(self):
        method = 'TH'
        mouse_video_mog = MouseVideo(self.test_video_mock_up, bkg_method='MOG')
        coords = mouse_video_mog.track_mouse()

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            cX, cY = coords[index]
            frame, roi =mouse_video_mog.calculate_roi(index, cX, cY, plot=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method MOG and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_track_mouse_TH_plotting(self):
        method = 'TH'
        coords = self.mouse_video.track_mouse()

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in [94]:  # range(self.mouse_video.num_frames):
            cX, cY = coords[index]
            print('x an y', cX, ', ', cY)
            frame, roi =self.mouse_video.calculate_roi(index, cX, cY, plot=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method MOG and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_track_mouse_MOG_plotting_crop(self):
        method = 'TH'
        mouse_video_mog = MouseVideo(self.test_video_mock_up, bkg_method='MOG')
        coords = mouse_video_mog.track_mouse()

        def gen():
            return random.randint(0, mouse_video_mog.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in [94]:  # range(self.mouse_video.num_frames):
            cX, cY = coords[index]
            print('index = ', index)
            print('x an y', cX, ', ', cY)

            frame, roi =mouse_video_mog.calculate_roi(index, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method MOG and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_track_mouse_TH_plotting_crop(self):
        method = 'TH'
        coords = self.mouse_video.track_mouse()

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)
        indices = [gen() for i in range(10)]
        for index in [94]:  # range(self.mouse_video.num_frames):
            print(index)
            cX, cY = coords[index]
            frame, roi =self.mouse_video.calculate_roi(index, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method TH and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_calculate_roi(self):
        method = self.mouse_video._bkg_method
        N,M,_ = self.mouse_video.frames[0].shape

        # coords = [(0,0),(10,10),(N-1,M-1), (N-1-100, M-1-100)]
        coords = [(0, 0), (10, 10), (N//2, M//2), (N - 1, M - 1), (N - 1 - 10, M - 1 - 10)]
        for cX, cY  in coords:
            frame, roi = self.mouse_video.calculate_roi(0, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'EVEN: this is the frame from the index {0} \n method TH and roi {roi}')
            plt.show()

        mouse_video_odd = MouseVideo(self.test_video_mock_up, roi_dims=(261, 261))
        N, M, _ = self.mouse_video.frames[0].shape

        for cX, cY in coords:
            frame, roi = mouse_video_odd.calculate_roi(0, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'ODD: this is the frame from the index {0} \n method TH and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

class TestMouseVideoMIT(TestCase):
    def setUp(self) -> None:
        self.test_video_mock_up = 'resources/mitmouse.mp4'
        if not os.path.exists(self.test_video_mock_up):
            self.test_video_mock_up = str(Path(os.path.join("..", self.test_video_mock_up)).resolve())
        print('path exists: ', os.path.exists(self.test_video_mock_up))
        self.mouse_video = MouseVideo(self.test_video_mock_up, bkg_method='MOG', roi_dims=(120,120))

    def test_detect_mouse(self):
        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        print('indices: ', indices)
        for index in range(self.mouse_video.num_frames):
            try:
                frame, roi = self.mouse_video.detect_mouse(index, plot=True)
                plt.imshow(frame)
                plt.title(f'this is the frame from the index {index} and roi{roi}')
                plt.show()
            except ValueError as e:
                print('Expected error at (test_detect_mouse) ', e)

    def test_detect_mouse_and_crop(self):
        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        for index in range(self.mouse_video.num_frames):
            try:
                frame, roi = self.mouse_video.detect_mouse(index, plot=True, crop=True)
                plt.imshow(frame)
                plt.title(f'this is the frame from the index {index}')
                plt.show()
            except ValueError as e:
                print('Expected error at (test_detect_mouse_and_crop) ', e)

    def test_get_no_background(self):

        method = self.mouse_video._bkg_method
        self.mouse_video._bkg_method = 'MOG'
        self.mouse_video.frames_no_bkg = None

        index = 10
        frame = self.mouse_video.frames_no_bkg[index]
        plt.imshow(frame)
        plt.title(f'this is the frame from the index {index}')
        plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_get_no_background_TH(self):
        method = self.mouse_video._bkg_method
        self.mouse_video._bkg_method = 'TH'
        self.mouse_video.frames_no_bkg = None

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            frame = self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_remove_darkchannel(self):
        blackchannel_imgs = self.mouse_video.remove_darkchannel()

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            frame = blackchannel_imgs[index]  # self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

    def test_track_mouse(self):
        self.mouse_video.frames_no_bkg = None
        coords = self.mouse_video.track_mouse()
        print(coords)
        plt.plot(coords)
        plt.show()

    def test_track_mouse_MOG(self):
        mouse_video_mog = MouseVideo(self.test_video_mock_up, bkg_method='MOG')
        coords = mouse_video_mog.track_mouse()
        print(coords)
        plt.plot(coords)
        plt.show()

    def test_track_mouse_MOG_plotting(self):
        method = 'TH'
        mouse_video_mog = self.mouse_video
        coords = mouse_video_mog.track_mouse()

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            cX, cY = coords[index]
            frame, roi =mouse_video_mog.calculate_roi(index, cX, cY, plot=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method MOG and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_track_mouse_large_ROI(self):
        method = 'TH'
        mouse_video_mog = MouseVideo(self.test_video_mock_up, bkg_method='MOG')
        coords = mouse_video_mog.track_mouse()

        def gen():
            return random.randint(0, mouse_video_mog.num_frames-1)

        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            cX, cY = coords[index]
            print('index = ', index)
            print('x an y', cX, ', ', cY)

            frame, roi =mouse_video_mog.calculate_roi(index, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method MOG and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_track_mouse_small_ROI(self):
        coords = self.mouse_video.track_mouse()
        method = self.mouse_video._bkg_method

        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)
        indices = [gen() for i in range(10)]
        for index in indices:  # range(self.mouse_video.num_frames):
            print(index)
            cX, cY = coords[index]
            frame, roi =self.mouse_video.calculate_roi(index, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} \n method TH and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_calculate_roi(self):
        method = self.mouse_video._bkg_method
        YY, XX, _ = self.mouse_video.frames[0].shape
        print('_, N,M', _, XX, YY)
        # coords = [(0,0),(10,10),(N-1,M-1), (N-1-100, M-1-100)]
        coords = [(0, 0), (10, 10), (XX//2, YY//2), (XX - 1, YY - 1), (XX - 1 - 10, YY - 1 - 10)]
        print('Coords: ', coords)
        for cX, cY  in coords:
            frame, roi = self.mouse_video.calculate_roi(0, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'EVEN: this is the frame from the index {0} \n method TH and roi {roi}')
            plt.show()

        mouse_video_odd = MouseVideo(self.test_video_mock_up, roi_dims=(261, 261))
        N, M, _ = self.mouse_video.frames[0].shape

        for cX, cY in coords:
            frame, roi = mouse_video_odd.calculate_roi(0, cX, cY, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'ODD: this is the frame from the index {0} \n method TH and roi {roi}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

