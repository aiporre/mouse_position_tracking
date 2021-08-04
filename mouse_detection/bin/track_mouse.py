from mouse_detection import MouseVideo
import argparse
from pathlib import Path


def track_mouse(directory='.', extension='avi', overwrite=False, fps=12, method='MOG', roi_dims=(260,260), remove_dark_channel=False):
    directory = Path(directory).resolve()

    for f in directory.rglob('*.' + extension):
        print('tracking in video : ', f)
        f = str(f)
        if f.endswith('-track.' + extension):
            continue
        mouse_video = MouseVideo(f, bkg_method=method, roi_dims=roi_dims)
        if remove_dark_channel:
            mouse_video.remove_darkchannel(inplace=True)

        if overwrite:
            mouse_video.save_rois(f, fps=fps)
        else:
            f_new = f.replace("." + extension, '-track.' + extension)
            mouse_video.save_rois(f_new, fps=fps)

def main():
    parser = argparse.ArgumentParser(description="Script to remove dark channel of video in directory")
    parser.add_argument('--directory', '-d', default='.',
                        help='Directory where to find the videos')
    parser.add_argument('--extension', '-e', default='avi',
                        help='Extension of the file that you need to find.')
    parser.add_argument('--replace', '-i', action='store_true',
                        help='Replace videos if permission is granted by user.')
    parser.add_argument('--fps', default=12,
                        help='Frame per second specification')
    parser.add_argument('--roi', default=(260,260), nargs=2, type=int,
                        help='roi dimensions')
    parser.add_argument('--method', default="MOG",
                        help='background subtraction method')
    parser.add_argument('--dark', action='store_true',
                        help='sets remove dark channel helps if the image is too dark')

    args = parser.parse_args()
    track_mouse(directory=args.directory, extension=args.extension, overwrite=args.replace, method=args.method, roi_dims=args.roi, remove_dark_channel=args.dark)

if __name__ == '__main__':
    main()
