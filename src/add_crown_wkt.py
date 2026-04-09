#!/usr/bin/env python
# coding: utf-8

"""
Docstring for add_crown_wkt.py

Aubrey Moore (aubreymoore2013@gmail.com)
Last modified: 2026-04-09

This python script segments a mask of a coconut palm tree detected by SAM3 into 2 parts: stem and crown.

Processing starts with a SQLite database with a 'detections' table containing a 'tree_wkt' which stores the tree mask for 
each detection as a polygon in Well Known Text (WKT) format.

The resulting crown mask is stored in the 'crown_wkt' field of the 'detections' table.
The crown mask can be further processed for detection of v-shaped cuts, a distinctive symptom of 
coconut rhinoceros beetle damage.

Documentation and visualization of the processing steps is being prepared. 
"""

import sqlite3
import numpy as np
import cv2
import roadside as rs
import textwrap
from icecream import ic


def contour2binary_image(image_height, image_width, contour):
    img = np.zeros((image_height, image_width), dtype=np.uint8)
    return cv2.drawContours(img, [contour], -1, (255), thickness=cv2.FILLED)


def gaussian_smooth(data, window_size):
    window = np.hanning(window_size) # Bell-shaped curve
    window /= window.sum()           # Normalize
    return np.convolve(data, window, mode='same')


def segment_crown(connection: sqlite3.Connection, detection_id: int) -> str:
    """ 
    Segements the crown of a coconut palm tree from the trunk.
    Processing starts with a SQLite database with a 'detections' table containing a 'tree_wkt' which stores the tree mask for 
    each detection as a polygon in Well Known Text (WKT) format.    
    A y-value, 'cut_line', which separates the crown from the stem is estimated.
    The 'crown_mask' is then segregated from the trunk and saved a polygon in WKT text format in the 'crown_wkt' field of the 
    detection record. 
    """
    
    # get data
    cursor = connection.cursor()
    sql = textwrap.dedent("""
        select image_height, image_width, tree_wkt
        from images, detections
        where 
            images.image_id=detections.image_id 
            AND detection_id = ?
        """)
    cursor.execute(sql, (detection_id,))
    image_height, image_width, tree_wkt = cursor.fetchone()

    # process data 
    
    # create a binary mask of the palm tree with the same shape as the original image
    tree_contour = rs.wkt2contour(tree_wkt)
    mask = contour2binary_image(image_height, image_width, tree_contour)

    # row_sums is the number of white pixels (value=1) in each row of the mask
    row_sums = np.sum(mask, axis=1)

    # normalized_row_sum is the proportion of mask pixels in each row
    total_row_sums = np.sum(row_sums, dtype=np.float64)
    normalized_row_sums = row_sums / total_row_sums

    # Calculate the difference between consecutive row sums to find where the sum changes significantly
    # to_begin=0 to keep the same length as row_sums
    differences = np.ediff1d(normalized_row_sums, to_begin=0) 
    gaussian_smoothed_differences = gaussian_smooth(differences, window_size=9)

    # cut_line estimates the y-value which separates the crown from the trunk
    # the 10 rows of pixels at the bottom of the mask are ignored when finding cut_line
    threshold = -0.00002
    stop_search = -10
    cut_line = np.max(np.where(gaussian_smoothed_differences[:stop_search] < threshold)).item()

    # segment the crown by removing nonzero pixels from cut_line and below
    # important: create a copy of mask instead of a view
    crown_mask = mask.copy()
    crown_mask[cut_line+1:, :] = 0

    # fit a polygon around the crown mask and convert to wkt text format for database storage
    contours, _ = cv2.findContours(crown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        crown_poly = np.squeeze(cv2.approxPolyDP(curve=cnt, epsilon=1, closed=True))
    crown_wkt = rs.conv_poly_from_array_to_wkt(crown_poly)
    cursor.close()
         
    return crown_wkt


# MAIN

db_path = 'test.db'
ic('starting')

# connect database
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# Fetch the data (ensure you include the Primary Key)
cursor.execute("SELECT detection_id FROM detections")
rows = cursor.fetchall()

# update the crown_wkt field for each row in the detections table
for row in rows:
    detection_id = row[0]
    ic(detection_id)
    crown_wkt = segment_crown(connection, detection_id)
    sql = f'UPDATE detections SET crown_wkt = "{crown_wkt}" WHERE detection_id = {detection_id}'
    cursor.execute(sql)

# commit changes and close database
connection.commit()
connection.close()

ic('finished')