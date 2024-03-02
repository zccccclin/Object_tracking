import cv2
import numpy as np
import time

class Keypoint_matcher():
    def __init__(self):
        self.orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

    def get_matchers(self, img1, img2, visualize):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        #Get good matches (nearby)
        if len(kp1) > 6 and len(kp2) > 6:
            matches = self.flann.knnMatch(des1,des2, k=2)
            good_matches = []
            il_good_matches =[]
            mask = np.array([])
            try:
                for m,n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
            #H, mask = pydegensac.findHomography(src_pts,dst_pts,4.0,0.99,2000)
            if len(src_pts) >4:
                H, mask = cv2.findHomography(src_pts,dst_pts,cv2.USAC_MAGSAC,5.0,0.99,2000)
            matchmask = mask.ravel().tolist()
            for m, n in zip(good_matches,matchmask):
                if n == 1:
                    il_good_matches.append(m)



        #Draw matches:
            if visualize:
                draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                    singlePointColor = None,
                    matchesMask = matchmask, # draw only inliers
                    flags = 2)
                self.img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
                cv2.drawMatches(img1, kp1, img2, kp2, good_matches, self.img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.drawMatches(img1, kp1, img2, kp2, good_matches, self.img_matches, **draw_params)
                #cv2.putText(self.img_matches, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(self.img_matches, f'No. Kp: {len(good_matches)}', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(self.img_matches, f'No. IL: {len(il_good_matches)}', (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Good Matches', self.img_matches)
                cv2.waitKey(0)

            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            return q1,q2
        else:
            return None, None
if __name__ == '__main__':
    vid = "videos/move1.mp4"

    #Video feed
    km = Keypoint_matcher()
    cap = cv2.VideoCapture(vid)
    process_frames = False
    old_frame = None
    new_frame = None
    frame_counter = 0
    while(cap.isOpened()):
        ret, new_frame = cap.read()
        frame_counter += 1
        start = time.perf_counter()
        if process_frames and ret:
            q1, q2 = km.get_matchers(old_frame,new_frame, True)
            #print(frame_counter)
        elif process_frames and ret is False:
            break
        old_frame = new_frame
        process_frames = True
    
        end = time.perf_counter()
        
        total_time = end - start
        fps = 1 / total_time
        
    cap.release()   

