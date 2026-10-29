
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def process_media(input_path: str, output_path: str) -> None:
    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    extension = source.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        image = cv2.imread(str(source))

        if image is None:
            raise ValueError("OpenCV could not decode this image.")

        destination.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(destination), image):
            raise RuntimeError("Could not save the output image.")

        print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
        print(f"Saved image: {destination}")

    elif extension in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(source))

        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        destination.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the output video.")

        frame_count = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                writer.write(frame)
                frame_count += 1
        finally:
            capture.release()
            writer.release()

        print(f"Video dimensions: {width} x {height}")
        print(f"FPS: {fps}")
        print(f"Frames processed: {frame_count}")
        print(f"Saved video: {destination}")

    else:
        raise ValueError(f"Unsupported file extension: {extension}")
