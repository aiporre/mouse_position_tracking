from mouse_detection import MouseVideo
import argparse
from pathlib import Path


def remove_background(directory='.', extension='avi', overwrite=False, fps=12, method="TH"):
    directory = Path(directory).resolve()

    for f in directory.rglob('*.' + extension):
        f = str(f)
        if f.endswith(f'-nobkg-{method}.{extension}'):
            continue
        mouse_video = MouseVideo(f, bkg_method=method)
        mouse_video.remove_background()
        if overwrite:
            mouse_video.save(f, fps=fps, no_background=True)
        else:
            f_new = f.replace("." + extension, f'-nobkg-{method}.{extension}')
            mouse_video.save(f_new, fps=fps, no_background=True)

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
    parser.add_argument('--method', default="TH", type=str,
                        help='Method for background subtraction')

    args = parser.parse_args()
    remove_background(directory=args.directory, extension=args.extension, overwrite=args.replace, method=args.method)

if __name__ == '__main__':
    main()
