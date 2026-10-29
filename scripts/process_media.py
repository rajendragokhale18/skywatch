
import argparse

from app.modules.media_io import process_media


def main():
    parser = argparse.ArgumentParser(
        description="Read and save an image or video."
    )
    parser.add_argument("input", help="Path to the input image or video")
    parser.add_argument("output", help="Path for the output image or video")

    args = parser.parse_args()
    process_media(args.input, args.output)


if __name__ == "__main__":
    main()
