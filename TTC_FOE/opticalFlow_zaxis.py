import numpy as np
import cv2
import time
import os

focal_length_mm = 3.67  
sensor_width_mm = 3.6  
image_width_pixels = 640  

focal_length_pixels = (focal_length_mm / sensor_width_mm) * image_width_pixels

calibration_file = 'camera_calibration.npz'
if os.path.exists(calibration_file):
    calibration_data = np.load(calibration_file)
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    focal_length = mtx[0, 0]  
    principal_point = (mtx[0, 2], mtx[1, 2])  
    print("Loaded calibration parameters.")
else:
    focal_length = focal_length_pixels
    principal_point = (320, 240) 
    print(f"Using approximate calibration parameters: focal_length = {focal_length:.2f} pixels")

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture(2) 
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./OUTPUT/output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))


def estimate_z_motion(foe, flow_magnitudes, focal_length):
    if foe is None or len(flow_magnitudes) == 0:
        print("FOE or flow magnitudes not valid")  
        return 0 

    avg_flow = np.mean(flow_magnitudes)
    z_motion = avg_flow / focal_length  
    print(f"Avg flow: {avg_flow}, Z-motion: {z_motion}")  
    return z_motion


def calculate_foe_ransac(p0, p1):
    flow = p1 - p0
    A = np.column_stack([flow[:, :, 0], flow[:, :, 1], np.ones_like(flow[:, :, 0])])
    _, _, V = np.linalg.svd(A)
    foe = V[-1, :-1] / V[-1, -1]

  
    foe_ransac, inliers = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
    return foe_ransac[:2] / foe_ransac[2]


def calculate_ttc(flow_magnitudes, focal_length, foe):
    
    distances = focal_length / (flow_magnitudes + 1e-6)  
    ttc = np.median(distances)  
    return ttc


while True:
    start = time.time()  
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        if len(trajectories) >= 8:
            foe = calculate_foe_ransac(p0, p1)
            flow_magnitudes = np.linalg.norm(p1 - p0, axis=2).flatten()
            print(f"Flow magnitudes: {flow_magnitudes}") 

            z_motion = estimate_z_motion(foe, flow_magnitudes, focal_length)
            ttc = calculate_ttc(flow_magnitudes, focal_length, foe)
            
            cv2.putText(img, f"Z-Motion: {z_motion:.2f}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(img, f"TTC: {ttc:.2f}", (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            #cv2.circle(img, (int(foe[0]), int(foe[1])), 5, (255, 0, 0), -1)  

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Optical Flow', img)
    out.write(img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
