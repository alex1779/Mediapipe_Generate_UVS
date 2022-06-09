# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:47:08 2022

@author: Ale
"""

import cv2
import numpy as np


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_landmark_points_face(image, face_landmarks):
    landmark_points_face = []
    height, width, _ = image.shape
    for index in range(468):
        x = int(face_landmarks.landmark[index].x * width)
        y = int(face_landmarks.landmark[index].y * height)
        landmark_points_face.append((x, y))
    return landmark_points_face


def get_landmark_points_UVS(face_landmarks, width, height):
    landmark_points_face = []
    for index in range(468):
        x = int(face_landmarks[index][0] * width)
        y = int(face_landmarks[index][1] * height)
        landmark_points_face.append((x, y))
    return landmark_points_face


def indexes_triangles_face(image, landmark_points_face):
    indexes_triangles = []
    height, width, _ = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    points = np.array(landmark_points_face, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points_face)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles





def sweep_faces(img, landmark_points_face, indexes_triangles, img2, landmark_points_2):
    try:

        img2_new_face = np.zeros_like(img2)
        points2 = np.array(landmark_points_2, np.int32)

        for triangle_index in indexes_triangles:

            tr1_pt1 = landmark_points_face[triangle_index[0]]
            tr1_pt2 = landmark_points_face[triangle_index[1]]
            tr1_pt3 = landmark_points_face[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)
            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            tr2_pt1 = landmark_points_2[triangle_index[0]]
            tr2_pt2 = landmark_points_2[triangle_index[1]]
            tr2_pt3 = landmark_points_2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2
            cropped_tr2_mask = np.zeros((h, w), np.uint8)
            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(
                img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(
                img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(
                img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

        return img2_new_face

    except:
        print('Cannot sweep face for this image.')
        return 0


def get_triangles(face_landmarks, indexes_triangles):
    triangles = []
    for idx, triangle in enumerate(indexes_triangles):
        PointA = face_landmarks[triangle[0]][0], face_landmarks[triangle[0]][1]
        PointB = face_landmarks[triangle[1]][0], face_landmarks[triangle[1]][1]
        PointC = face_landmarks[triangle[2]][0], face_landmarks[triangle[2]][1]
        triangles.append([PointA, PointB, PointC])
    return triangles


def draw_triangles(image, triangles, color=(255, 255, 255), thickness=1):
    for idx, triangle in enumerate(triangles):
        cv2.line(image, triangle[0], triangle[1], color, thickness)
        cv2.line(image, triangle[1], triangle[2], color, thickness)
        cv2.line(image, triangle[2], triangle[0], color, thickness)




