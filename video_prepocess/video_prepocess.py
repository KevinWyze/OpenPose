import sys
sys.path.append('../')
import cv2
from openpose.body.estimator import BodyPoseEstimator
#from openpose.utils import draw_body_connections, draw_keypoints
import pandas as pd

class VideoPreprocess:
    def __init__(self):
        self.estimator = BodyPoseEstimator(pretrained=True)


    def preprocess(self, videoClip):
        self.calc_timestamps = [0.0]
        fps = videoClip.get(cv2.CAP_PROP_FPS)
        time_stamps = [videoClip.get(cv2.CAP_PROP_POS_MSEC)]
        kp_ts = []

        while videoClip.isOpened():
            flag, frame = videoClip.read()
            if not flag:
                break
            time_stamps.append(videoClip.get(cv2.CAP_PROP_POS_MSEC))
            self.calc_timestamps.append(self.calc_timestamps[-1] + 1000 / fps)
            kp = self.estimator(frame)
            # print('time')
            # print(keypoints)
            # frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
            # frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
            kp_store = kp.copy()
            kp_ts.append(kp_store)
            cv2.imshow('Video Demo', frame)
            if cv2.waitKey(20) & 0xff == 27:  # exit if pressed `ESC`
                break
        videoClip.release()
        cv2.destroyAllWindows()
        return kp_ts

    def preparedata(self, videoclip, kp_ts):
        data = dict()
        data['x_position'] = []
        data['y_position'] = []
        data['point_id'] = []
        data['timestamp'] = []
        data['subject_id'] = []

        for i in range(len(kp_ts)):
            temp = kp_ts[i]
            subject_num = len(temp)
            for j in range(subject_num):
                for k in range(len(temp[j])):
                    data['x_position'].append(temp[j][k][0])
                    data['y_position'].append(temp[j][k][1])
                    data['point_id'].append(k + 1)
                    data['timestamp'].append(self.calc_timestamps[i])
                    data['subject_id'].append(j + 1)

        df = pd.DataFrame.from_dict(data)
        data_name = 'name' + '.csv'
        df.to_csv(data_name)


if __name__ == '__main__':
    VP = VideoPreprocess()
    videoClip = cv2.VideoCapture('../media/example.mp4')
    kp_ts = VP.preprocess(videoClip)
    VP.preparedata(videoClip, kp_ts)







