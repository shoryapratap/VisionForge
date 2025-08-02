        from ultralytics import YOLO
        from pathlib import Path
        import cv2
        import os
        import yaml
        import sys

        def predict_and_save(model, image_path, output_img_path, output_label_path):
            results = model.predict(image_path, conf=0.5)
            result = results[0]
            annotated_img = result.plot()
            cv2.imwrite(str(output_img_path), annotated_img)
            with open(output_label_path, 'w') as f:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    x_center, y_center, width, height = box.xywh[0].tolist()
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

        if __name__ == '__main__':
            this_dir = Path(__file__).resolve().parent
            os.chdir(this_dir)

            yaml_path = this_dir / 'data' / 'yolo_params.yaml'
            if not yaml_path.exists():
                print(f"❌ ERROR: yolo_params.yaml not found at {yaml_path}")
                sys.exit(1)

            with open(yaml_path, 'r') as file:
                data_cfg = yaml.safe_load(file)
            test_dir = data_cfg.get('path')
            images_dir = Path(test_dir) / 'test' / 'images'
            
            image_list = list(images_dir.glob("*.[jpJP]*[npNP]*[geGE]*"))
            if not image_list:
                print(f"⚠️ No images found in {images_dir}")
                sys.exit(1)

            detect_path = this_dir / "runs" / "detect"
            train_dirs = [d for d in detect_path.iterdir() if d.is_dir()]
            if not train_dirs:
                print("❌ ERROR: No trained model folders found in runs/detect/")
                sys.exit(1)

            latest_train = max(train_dirs, key=os.path.getmtime)
            model_path = latest_train / "weights" / "best.pt"
            if not model_path.exists():
                print(f"❌ ERROR: best.pt not found at {model_path}")
                sys.exit(1)

            print(f"✅ Using model weights: {model_path}")
            model = YOLO(str(model_path))

            output_base = this_dir / "predictions"
            img_output_dir = output_base / "images"
            lbl_output_dir = output_base / "labels"
            img_output_dir.mkdir(parents=True, exist_ok=True)
            lbl_output_dir.mkdir(parents=True, exist_ok=True)

            print("🚀 Starting predictions...")
            for img_path in image_list:
                output_img = img_output_dir / img_path.name
                output_lbl = lbl_output_dir / (img_path.stem + ".txt")
                predict_and_save(model, img_path, output_img, output_lbl)
            
            print("\n📊 Running evaluation on test set...")
            model.val(data=str(yaml_path), split="test")