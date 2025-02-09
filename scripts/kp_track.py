from pickle import TRUE
import numpy as np
import cv2
import time

lk_params = dict(winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                qualityLevel = 0.3,
                minDistance = 10,
                blockSize = 7 )


trajectory_len = 10
detect_interval = 5
trajectories = []
frame_idx = 0

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
            table_number = 6, # 12
            key_size = 12,     # 20
            multi_probe_level = 1) #2
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)

cap = cv2.VideoCapture('videos/rotation_occlusion.mp4')
detector = cv2.ORB_create()
total_track = 0
while True:

    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()
    if not suc:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), zip(good,_st.flatten())):
            good_flag = (good_flag[0] and bool(good_flag[1]))
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)

        trajectories = new_trajectories

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (255, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
        #total_track += len(trajectories)
        #cv2.putText(img, 'avg track: %d' % (total_track/frame_idx), (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        
        p = detector.detect(frame_gray,mask)
        p = [[(kp.pt[0], kp.pt[1])] for kp in p]
        p = np.array(p)
        print(p.shape)
        if p is not None:
        # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])
    if frame_idx == 0:
        #out = cv2.VideoWriter('result/trial.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 24, (int(cap.get(3)),int(cap.get(4))))
        pass
    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    
    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Optical Flow', img)
    #out.write(img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
#out.release()
cap.release()
cv2.destroyAllWindows()