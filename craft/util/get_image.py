import os
import cv2


source_dir = "/home/lichengzhi/CRAFT-pytorch/demo"
target_dir = "/home/lichengzhi/CRAFT-pytorch/data"


def main():
    for r, _, files in os.walk(source_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                cv2.imwrite(os.path.join(target_dir, file), img)


if __name__ == "__main__":
    main()
