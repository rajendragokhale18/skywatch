
from pathlib import Path

import cv2
import numpy as np


OUTPUT_DIR = Path("data/videos")
WIDTH = 640
HEIGHT = 480
FPS = 20
FRAME_COUNT = 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a blue background image.
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = (180, 120, 50)  # BGR color

    # Draw a white circular object.
    cv2.circle(image, (320, 240), 12, (255, 255, 255), -1)

    image_path = OUTPUT_DIR / "sample.jpg"
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError("Could not save test image.")

    # Create a video containing the moving object.
    video_path = OUTPUT_DIR / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create test video.")

    try:
        for frame_index in range(FRAME_COUNT):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[:] = (180, 120, 50)

            x = 40 + frame_index * 5
            y = HEIGHT // 2

            cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Created image: {image_path}")
    print(f"Created video: {video_path}")
    print(f"Video duration: {FRAME_COUNT / FPS:.1f} seconds")


if __name__ == "__main__":
    main()
