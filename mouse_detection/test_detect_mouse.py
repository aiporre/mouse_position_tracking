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
            self.test_video_mock_up = str(Path(os.path.join("..",self.test_video_mock_up)).resolve())
        print('path exists: ', os.path.exists(self.test_video_mock_up))
        self.mouse_video = MouseVideo(self.test_video_mock_up, bkg_method='TH')


    def test_detect_mouse(self):
        # index =96
        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        for index in indices: #range(self.mouse_video.num_frames):
            frame, roi = self.mouse_video.detect_mouse(index, plot=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index} and roi{roi}')
            plt.show()

    def test_detect_mouse_and_crop(self):
        # index =96
        def gen():
            return random.randint(0, self.mouse_video.num_frames - 1)

        indices = [gen() for i in range(10)]
        for index in indices: #range(self.mouse_video.num_frames):
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
        for index in indices: #range(self.mouse_video.num_frames):
            frame, roi = mouse_video_mog.detect_mouse(index, plot=True, crop=True)
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

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
            return random.randint(0, self.mouse_video.num_frames)

        indices = [gen() for i in range(10)]
        for index in indices: #range(self.mouse_video.num_frames):
            frame = self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None

    def test_remove_darkchannel(self):
        blackchannel_imgs = self.mouse_video.remove_darkchannel()
        def gen():
            return random.randint(0, self.mouse_video.num_frames)
        indices = [gen() for i in range(10)]
        for index in indices: #range(self.mouse_video.num_frames):
            frame = blackchannel_imgs[index] # self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

