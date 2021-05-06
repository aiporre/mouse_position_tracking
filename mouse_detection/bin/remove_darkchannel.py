from mouse_detection import MouseVideo
import argparse
from pathlib import Path


def remove_darkchannel(directory='.', extension='avi', overwrite=False, fps=12):
    directory = Path(directory).resolve()

    for f in directory.rglob('*.' + extension):
        f = str(f)
        if f.endswith('-dark.' + extension):
            continue
        mouse_video = MouseVideo(f, bkg_method='TH')
        mouse_video.remove_darkchannel(inplace=True)
        if overwrite:
            mouse_video.save(f, fps=fps)
        else:
            f_new = f.replace("." + extension, '-dark.' + extension)
            mouse_video.save(f_new, fps=fps)

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

    args = parser.parse_args()
    remove_darkchannel(directory=args.directory, extension=args.extension, overwrite=args.replace)

if __name__ == '__main__':
    main()
