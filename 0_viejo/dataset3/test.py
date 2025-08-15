from ultralytics import YOLO

# Run inference on an image
# yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt

model = YOLO('yolo11l.pt')  
results = model("test2.jpeg") # results list

# Process results list 
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probabilities object for classification outputs
    obb = result.obb  # OBB object for rotated bbox outputs
    result.show()  # Display results
    result.save(filename="test2.jpg")  # Save results to fileP