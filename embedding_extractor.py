"""
Extract 128D face embeddings from enrolled face images and save to CSV.
Run after face_enrollment.py; output is used by realtime_recognizer.py.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import dlib
import csv
import numpy as np
import logging
import cv2

from config.paths import (
    FACES_CAPTURE_DIR,
    FEATURES_CSV_PATH,
    SHAPE_PREDICTOR_PATH,
    FACE_RECOGNITION_MODEL_PATH,
)

path_images_from_camera = str(FACES_CAPTURE_DIR) + os.sep

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
face_reco_model = dlib.face_recognition_model_v1(str(FACE_RECOGNITION_MODEL_PATH))


def compute_128d_features(path_img):
    """Compute 128D face descriptor for the first face found in the image."""
    img_rd = cv2.imread(path_img)
    if img_rd is None:
        logging.warning("Could not read image: %s", path_img)
        return 0
    faces = detector(img_rd, 1)
    logging.info("%-40s %-20s", "Image with face detected:", path_img)
    if len(faces) == 0:
        logging.warning("No face in image")
        return 0
    shape = predictor(img_rd, faces[0])
    return face_reco_model.compute_face_descriptor(img_rd, shape)


def mean_embeddings_for_person(person_dir):
    """Compute mean 128D embedding across all face images in a person's folder."""
    features_list = []
    if not os.path.isdir(person_dir):
        logging.warning("Not a directory: %s", person_dir)
        return np.zeros(128, dtype=object, order='C')
    photos = os.listdir(person_dir)
    for photo in photos:
        path = os.path.join(person_dir, photo)
        if not os.path.isfile(path):
            continue
        logging.info("%-40s %-20s", "Reading image:", path)
        feat = compute_128d_features(path)
        if feat != 0:
            features_list.append(feat)
    if features_list:
        return np.array(features_list, dtype=object).mean(axis=0)
    return np.zeros(128, dtype=object, order='C')


def main():
    logging.basicConfig(level=logging.INFO)
    if not FACES_CAPTURE_DIR.exists():
        logging.warning("Enrollment directory not found. Run face_enrollment.py first.")
        return
    person_list = sorted(os.listdir(path_images_from_camera))
    with open(str(FEATURES_CSV_PATH), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            person_path = path_images_from_camera + person
            if not os.path.isdir(person_path):
                continue
            logging.info("%s person: %s", path_images_from_camera, person)
            features_mean = mean_embeddings_for_person(person_path)
            if len(person.split('_', 2)) == 2:
                person_name = person
            else:
                person_name = person.split('_', 2)[-1]
            row = np.insert(features_mean, 0, person_name, axis=0)
            writer.writerow(row)
    logging.info("Saved face embeddings to: %s", FEATURES_CSV_PATH)


if __name__ == '__main__':
    main()
