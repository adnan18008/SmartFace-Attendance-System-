"""
Real-time face recognition and attendance logging.
Uses Dlib ResNet-based 128D embeddings and centroid tracking for multi-face recognition.
"""
import sys
from pathlib import Path

# Ensure project root is on path when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

from config.paths import (
    SHAPE_PREDICTOR_PATH,
    FACE_RECOGNITION_MODEL_PATH,
    FEATURES_CSV_PATH,
    DATABASE_PATH,
)

# Dlib frontal face detector
detector = dlib.get_frontal_face_detector()
# 68-point face landmark predictor
predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
# ResNet-based 128D face descriptor model
face_reco_model = dlib.face_recognition_model_v1(str(FACE_RECOGNITION_MODEL_PATH))

# Initialize attendance database
conn = sqlite3.connect(str(DATABASE_PATH))
cursor = conn.cursor()
current_date = datetime.datetime.now().strftime("%Y_%m_%d")
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)
conn.commit()
conn.close()


class RealtimeFaceRecognizer:
    """Real-time face recognition with centroid tracking and automatic attendance logging."""

    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        self.frame_cnt = 0

        self.face_features_known_list = []
        self.face_name_known_list = []
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    def load_face_database(self):
        """Load known face embeddings from features CSV."""
        if FEATURES_CSV_PATH.exists():
            csv_rd = pd.read_csv(str(FEATURES_CSV_PATH), header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    features_someone_arr.append(
                        '0' if csv_rd.iloc[i][j] == '' else csv_rd.iloc[i][j]
                    )
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in database: %d", len(self.face_features_known_list))
            return True
        logging.warning("Features CSV not found. Run face_enrollment.py then embedding_extractor.py first.")
        return False

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def euclidean_distance(feature_1, feature_2):
        """Compute L2 distance between two 128D face embeddings."""
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        return float(np.sqrt(np.sum(np.square(feature_1 - feature_2))))

    def centroid_tracker(self):
        """Link faces in current frame to identities from previous frame via centroid proximity."""
        for i in range(len(self.current_frame_face_centroid_list)):
            distances = []
            for j in range(len(self.last_frame_face_centroid_list)):
                d = self.euclidean_distance(
                    self.current_frame_face_centroid_list[i],
                    self.last_frame_face_centroid_list[j],
                )
                distances.append(d)
            best_idx = distances.index(min(distances))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[best_idx]

    def draw_overlay(self, img_rd):
        """Draw FPS, frame count, and face count on the frame."""
        cv2.putText(
            img_rd, "Face Recognizer with Deep Learning",
            (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(round(self.fps, 2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        for i in range(len(self.current_frame_face_name_list)):
            cx, cy = int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])
            cv2.putText(img_rd, "Face_" + str(i + 1), (cx, cy), self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def log_attendance(self, name):
        """Record attendance for the given identity (one entry per person per day)."""
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        if cursor.fetchone():
            logging.info("%s already marked present for %s", name, current_date)
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            logging.info("%s marked present for %s at %s", name, current_date, current_time)
        conn.close()

    def process_stream(self, stream):
        """Main recognition loop: detect faces, compute embeddings, match to database, log attendance."""
        if not self.load_face_database():
            return
        while stream.isOpened():
            self.frame_cnt += 1
            logging.debug("Frame %s starts", self.frame_cnt)
            _, img_rd = stream.read()
            kk = cv2.waitKey(1)

            faces = detector(img_rd, 0)
            self.last_frame_face_cnt = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)
            self.last_frame_face_name_list = self.current_frame_face_name_list[:]
            self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []

            if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                    self.reclassify_interval_cnt != self.reclassify_interval):
                self.current_frame_face_position_list = []
                if "unknown" in self.current_frame_face_name_list:
                    self.reclassify_interval_cnt += 1
                if self.current_frame_face_cnt != 0:
                    for k, d in enumerate(faces):
                        self.current_frame_face_position_list.append((
                            faces[k].left(),
                            int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4),
                        ))
                        self.current_frame_face_centroid_list.append([
                            (faces[k].left() + faces[k].right()) / 2,
                            (faces[k].top() + faces[k].bottom()) / 2,
                        ])
                        cv2.rectangle(
                            img_rd, (d.left(), d.top()), (d.right(), d.bottom()),
                            (255, 255, 255), 2
                        )
                if self.current_frame_face_cnt != 1:
                    self.centroid_tracker()
                for i in range(self.current_frame_face_cnt):
                    cv2.putText(
                        img_rd, self.current_frame_face_name_list[i],
                        self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA
                    )
                self.draw_overlay(img_rd)
            else:
                self.current_frame_face_position_list = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list = []
                self.reclassify_interval_cnt = 0
                if self.current_frame_face_cnt == 0:
                    self.current_frame_face_name_list = []
                else:
                    self.current_frame_face_name_list = []
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape)
                        )
                        self.current_frame_face_name_list.append("unknown")
                    for k in range(len(faces)):
                        self.current_frame_face_centroid_list.append([
                            (faces[k].left() + faces[k].right()) / 2,
                            (faces[k].top() + faces[k].bottom()) / 2,
                        ])
                        self.current_frame_face_X_e_distance_list = []
                        self.current_frame_face_position_list.append((
                            faces[k].left(),
                            int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4),
                        ))
                        for i in range(len(self.face_features_known_list)):
                            if str(self.face_features_known_list[i][0]) != '0.0':
                                d = self.euclidean_distance(
                                    self.current_frame_face_feature_list[k],
                                    self.face_features_known_list[i],
                                )
                                self.current_frame_face_X_e_distance_list.append(d)
                            else:
                                self.current_frame_face_X_e_distance_list.append(999999999)
                        min_dist = min(self.current_frame_face_X_e_distance_list)
                        similar_idx = self.current_frame_face_X_e_distance_list.index(min_dist)
                        if min_dist < 0.4:
                            name = self.face_name_known_list[similar_idx]
                            self.current_frame_face_name_list[k] = name
                            self.log_attendance(name)
                    self.draw_overlay(img_rd)

            if kk == ord('q'):
                break
            self.update_fps()
            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)
        stream.release()
        cv2.destroyAllWindows()

    def run(self, camera_id=0):
        """Start real-time recognition from the default camera."""
        cap = cv2.VideoCapture(camera_id)
        self.process_stream(cap)


def main():
    logging.basicConfig(level=logging.INFO)
    recognizer = RealtimeFaceRecognizer()
    recognizer.run()


if __name__ == '__main__':
    main()
