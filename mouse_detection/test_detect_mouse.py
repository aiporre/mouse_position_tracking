from unittest import TestCase
from .detect_mouse import MouseVideo
import matplotlib.pyplot as plt
import random

class TestMouseVideo(TestCase):
    def setUp(self) -> None:
        self.mouse_video = MouseVideo('resouces/mouse_cut.avi', bkg_method='TH')

    def test_detect_mouse(self):
        # index =96
        def gen():
            return random.randint(0, self.mouse_video.num_frames-1)
        indices = [gen() for i in range(10)]
        for index in range(self.mouse_video.num_frames):
            frame = self.mouse_video.detect_mouse(index, plot=True)
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
        for index in range(self.mouse_video.num_frames):
            frame = self.mouse_video.frames_no_bkg[index]
            plt.imshow(frame)
            plt.title(f'this is the frame from the index {index}')
            plt.show()

        self.mouse_video._bkg_method = method
        self.mouse_video.frames_no_bkg = None


