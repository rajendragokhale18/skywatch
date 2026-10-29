
import argparse

from app.modules.motion_candidates import detect_motion_video


def main():
    parser = argparse.ArgumentParser(
        description="Detect moving regions in a video."
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument(
        "--min-area",
        type=int,
        default=80,
        help="Minimum contour area in pixels",
    )

    args = parser.parse_args()

    detect_motion_video(
        input_path=args.input,
        output_path=args.output,
        min_area=args.min_area,
    )


if __name__ == "__main__":
    main()
