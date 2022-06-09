import cv2
import numpy as np
from datetime import datetime
from data.face_swap import *
# from data.drawing import detect_side_face_minimal
from data.UVS import UVS


def save_image(img):
    timestr = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
    cv2.imwrite('output/'+timestr+'.jpg', img)


def generate_uvs(image, landmark_list, width, height):
    landmark_points = get_landmark_points_face(image, landmark_list)
    indexes_triangles = indexes_triangles_face(image, landmark_points)
    img2 = np.zeros([width, height, 3], dtype=np.uint8)
    landmark_points2 = get_landmark_points_UVS(UVS, width, height)
    indexes_triangles2 = indexes_triangles_face(img2, landmark_points2)
    uvs_face = sweep_faces(image, landmark_points,
                            indexes_triangles, img2, landmark_points2)
    triangles = get_triangles(landmark_points2, indexes_triangles2)
    # drawTesselation(uvs_face, triangles)
    
    
    return uvs_face


def drawTesselation(image, triangles):

    draw_triangles(image, triangles, color=(255, 255, 255), thickness=1)
        
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    