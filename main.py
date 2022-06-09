# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:15:23 2022

@author: Ale
"""

import argparse
import cv2
import mediapipe as mp
import imutils
from data.centerface import generate_uvs, save_image

mp_face_mesh = mp.solutions.face_mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='input/01.jpg',
                        help='Please specify path for image', required=True)
    opt = parser.parse_args()
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        while True:
            image = cv2.imread(opt.img_path)
            
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
            
            for face_landmarks in results.multi_face_landmarks:
    
                image_uvs = generate_uvs(
                    image=image,
                    landmark_list=face_landmarks,
                    width=1500,
                    height=1500,
                )
    
            annotated_image = image_uvs.copy()
            
            cv2.putText(annotated_image, "Press S to save", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            
            annotated_image = imutils.resize(annotated_image, height=600)
            
            cv2.imshow('UVS Generated', annotated_image)
            keypressed = cv2.waitKey(0)
    
            if keypressed == 27:
                cv2.destroyAllWindows()
                break
    
            if keypressed == ord('s'):
                print('Image Saved!')
                save_image(image_uvs)
                saved = True


if __name__ == "__main__":
    main()
    


    
    
    
    
    
    
    
    
    
    
    
    