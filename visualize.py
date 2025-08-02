import os
import cv2

class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2

    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        classes_file = os.path.join(dataset_folder, "classes.txt")
        with open(classes_file, "r") as f:
            self.classes = {i: c for i, c in enumerate(f.read().splitlines())}
        self.set_mode(self.MODE_TRAIN)

    def set_mode(self, mode=MODE_TRAIN):
        self.mode = mode
        if mode == self.MODE_TRAIN:
            split = "train"
        elif mode == self.MODE_VAL:
            split = "val"
        else:
            split = "test"
        
        self.images_folder = os.path.join(self.dataset_folder, split, "images")
        self.labels_folder = os.path.join(self.dataset_folder, split, "labels")

        self.image_names = sorted([f for f in os.listdir(self.images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.label_names = sorted([f for f in os.listdir(self.labels_folder) if f.endswith('.txt')])
        self.num_images = len(self.image_names)

        if self.num_images == 0:
            print(f"❌ No images found in {self.images_folder}")
            exit(1)

        self.frame_index = 0
        print(f"🔄 Switched to mode: {split.upper()} | Images: {self.num_images}")

    def next_frame(self):
        self.frame_index = (self.frame_index + 1) % self.num_images

    def previous_frame(self):
        self.frame_index = (self.frame_index - 1 + self.num_images) % self.num_images

    def draw_labels(self, image, label_path):
        try:
            with open(label_path, "r") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            return image  # No label file, return original image

        for line in lines:
            if not line.strip(): continue
            parts = line.split()
            if len(parts) < 5:
                continue  # Malformed line

            class_index, x, y, w, h = map(float, parts[:5])
            cx = int(x * image.shape[1])
            cy = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])
            x1 = cx - w // 2
            y1 = cy - h // 2

            color = (0, 255, 0)
            label = self.classes.get(int(class_index), f"Class {int(class_index)}")
            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image

    def seek_frame(self, idx):
        image_file = os.path.join(self.images_folder, self.image_names[idx])
        label_file = os.path.join(self.labels_folder, os.path.splitext(self.image_names[idx])[0] + '.txt')

        image = cv2.imread(image_file)
        if image is None:
            image = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "Image Load Failed", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image

        image = self.draw_labels(image, label_file)
        info = f"[{self.frame_index+1}/{self.num_images}] {self.image_names[idx]} - Mode: {['Train','Val','Test'][self.mode]}"
        cv2.putText(image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        instructions = "← prev | → next | T/V/E: mode | Q: quit"
        cv2.putText(image, instructions, (10, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return image

    def run(self):
        print("📸 Press [←] or [→] to navigate. Press 't', 'v', 'e' to switch dataset. Press 'q' to quit.")
        while True:
            frame = self.seek_frame(self.frame_index)
            frame = cv2.resize(frame, (960, 720))
            cv2.imshow("YOLO Visualizer", frame)
            key = cv2.waitKey(0)

            if key in [27, ord('q')]:
                break
            elif key in [81, ord('a')]:  # Left arrow / A
                self.previous_frame()
            elif key in [83, ord('d')]:  # Right arrow / D
                self.next_frame()
            elif key in [ord('t'), ord('T')]:
                self.set_mode(self.MODE_TRAIN)
            elif key in [ord('v'), ord('V')]:
                self.set_mode(self.MODE_VAL)
            elif key in [ord('e'), ord('E')]:
                self.set_mode(self.MODE_TEST)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vis = YoloVisualizer(os.path.join(os.path.dirname(__file__), "data"))
    vis.run()
