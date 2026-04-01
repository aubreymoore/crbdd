#!/usr/bin/env python
# coding: utf-8

"""
Docstring for roadside.py

Aubrey Moore (aubreymoore2013@gmail.com)
Last modified: 2026-02-19

This module provides python functions for building automated detection of coconut rhinoceros beetle 
damage in digital images.

This package requires `sam3.pt` which contains weights for the SAM3 model.
You must manually request access via `Hugging Face`.
Once approved, download the file and place it in your working directory.
`sam3.pt` is a large file, so include it in your `.gitignore`.

If you are using `uv` to manage your python environment and dependencies, use this command to keep dependencies up to date:
```
uv sync --upgrade
```

Run test with:
```
uv run -m pytest -s -v src/roadside_test.py
```
"""

from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import numpy as np
import torch
import os
import exif
import pandas as pd
import sqlite3
from pathlib import Path
import gc
from time import sleep
import shapely
from shapely.geometry import Polygon
from shapely import wkt
from shapely.affinity import affine_transform
from shapely.wkt import loads as wkt_loads

import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.spatial.distance import euclidean
from pathlib import Path
from icecream import ic
import textwrap
import math


def flip_wkt_origin(wkt_string:str, image_height:int) -> str:
    """
    Transforms a WKT polygon from top-left origin to bottom-left origin.
    
    Formula:
    x' = x
    y' = height - y
    """
    # Load the polygon from the WKT string
    poly = wkt.loads(wkt_string)
    
    # Define the transformation matrix (affine_transform)
    # [a, b, d, e, xoff, yoff] 
    # represents:
    # x' = ax + by + xoff
    # y' = dx + ey + yoff
    # To get y' = -y + height, we set e = -1 and yoff = image_height
    matrix = [1, 0, 0, -1, 0, image_height]
    
    transformed_poly = affine_transform(poly, matrix)
    
    return transformed_poly.wkt

# --- Example Usage ---
# image_h = 1000
# original_wkt = "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))"

# new_wkt = flip_wkt_origin(original_wkt, image_h)

# print(f"Original:  {original_wkt}")
# print(f"Corrected: {new_wkt}")



def conv_poly_from_array_to_wkt(poly: np.array) -> str:
    return Polygon(poly).wkt

# # Usage example:
#
# poly_wkt = 'POLYGON((0 0, 0 40, 40 40, 40 0, 0 0))'
# poly = conv_poly_from_wkt_to_array(poly_wkt)
# ic(conv_poly_from_array_to_wkt(poly));


def conv_poly_from_wkt_to_array(poly_wkt: str) -> np.array:
    return np.array(shapely.from_wkt(poly_wkt).exterior.coords, dtype=np.int32)

# # Usage example:

# poly_wkt = 'POLYGON((0 0, 0 40, 40 40, 40 0, 0 0))'
# ic(conv_poly_from_wkt_to_array(poly_wkt));


def check_gpu():
    """ 
    Checks for GPU availability and prints CUDA version and GPU device name if available.
    Returns True if GPU is available, otherwise False.
    """
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU available.")
        return False


def run_sam3_semantic_predictor(input_image_path: str, text_prompts: list=['coconut palm tree']) -> list:
    """ 
    Uses the SAM3 semantic predictor to detect objects specified by text prompts in an image.
    
    Inputs:
      input_image_path relative to working directory 
      text_prompts: list of text prompts; default: ['coconut palm tree']
      
    Outputs:
      results:     
    """
    # Initialize predictor with configuration
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,  # Use FP16 for faster inference
        save=True,  # Save image visualizing output results
        save_txt=False,  # Save output results in text format
        save_conf=False,  # Save confidence scores   
        imgsz=1932,  # Adjusted image size from 1920 to meet stride 14 requirement
        batch=1,
        device="0",  # Use GPU device 0
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Set image once for multiple queries
    predictor.set_image(input_image_path)

    # Query with multiple text prompts
    results = predictor(text=text_prompts)

    return results

## Example usage:
#
# root_dir = "/home/aubrey/Desktop/sam3-2026-01-31"
# image_paths = ["20251129_152106.jpg", "08hs-palms-03-zglw-superJumbo.webp"]
# text_prompts = ["coconut palm tree"]
#
# os.chdir(root_dir) # ensure we start in the correct directory
# for image_path in image_paths:
#     results_gpu = run_sam3_semantic_predictor(image_path, text_prompts)
#
#     # Free up GPU memory in preparation for detecting objects in the next image
#     # This is a work-around to prevent out-of-memory errors from the GPU
#     # I move all results for further processing and use the GPU only for object detection.
#     print('deleting results from GPU memory')       
#     results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
#     delete_results_from_gpu_memory()
#
# print("Processing complete.")
    
    
def get_data_for_images_table(results_cpu, image_path: str) -> pd.DataFrame:
    """ 
    Gets data for for a single image for insertion as a record in the 'images' database table. 
    Returns a Pandas dataframe containing image_path, image_width, image_height, timestamp, latitude, longitude)
    image_width and image_height come from results_cpu
    timestamp, latitude, longitude come from the EXIF metadata embedded in the image, if it exists. 
    """

    image_height = results_cpu[0].orig_shape[0]
    image_width = results_cpu[0].orig_shape[1]

    with open(image_path, 'rb') as f:
        imgx = exif.Image(f)

    if imgx.has_exif:
        # to see all available exif_data use imgx.get_all()
        
        # timestamp
        timestamp = imgx.datetime
            
        # latitude
        d, m, s = imgx.gps_latitude
        latitude = d + m/60 + s/3600   
        if imgx.gps_latitude_ref == 'S':
            latitude = -latitude  

        # longitude
        d, m, s = imgx.gps_longitude
        longitude = d + m/60 + s/3600   
        if imgx.gps_longitude_ref == 'W':
            longitude = -longitude
        longitude
    else:
        timestamp = pd.NA
        latitude = pd.NA
        longitude= pd.NA
        
    df = pd.DataFrame({
        'image_path': image_path,
        'image_width': image_width,
        'image_height': image_height,
        'timestamp': timestamp,
        'latitude': latitude,
        'longitude': longitude
    },index=[0])
    
    return df

## Usage example:
#
# image_path = image_paths[0]
# results_gpu = rs.run_sam3_semantic_predictor(input_image_path=image_path, text_prompts=text_prompts)
# results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
# _results_from_gpu_memory() # Clear GPU memory after processing each image
# get_data_for_images_table(results_cpu)


def get_data_for_detections_table(results_cpu, image_id:int)->pd.DataFrame:
    """ 
    Gets data for for a single image for insertion as records in the 'detections' database table. 
    Returns data as a pandas dataframe containing columns for image_id, class_id, poly_wkt, x_min, y_min, x_max, y_max, confidence
    """

    # Process detection results (assuming one image for simplicity: results[0])
    result = results_cpu[0]

    # create a pandas dataframe for bounding boxes
    boxes_data = result.boxes.data.tolist()
    df_boxes = pd.DataFrame(boxes_data, columns=['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id'])

    # create a pandas dataframe for segmentation masks (polygons)
    masks_data = []
    # Iterate over each detected object's mask
    for i, mask in enumerate(result.masks.xy):
        poly_arr = mask
        poly_wkt = conv_poly_from_array_to_wkt(poly_arr)
        poly_wkt_c = flip_wkt_origin(poly_wkt, image_height=result.orig_shape[0]) # correct for coordinate system origin
        
        
        # ic(poly_arr)
        hull_indices = cv2.convexHull(poly_arr, returnPoints=False)
        # ic(hull_indices)
        hull_indices_string = np_int_array_to_string(hull_indices)


        masks_data.append({
            # 'image_path': image_path,
            # 'object_index': i, 
            'class_id': df_boxes.iloc[i]['class_id'], 
            'poly_wkt': poly_wkt,
            'poly_wkt_c': poly_wkt_c,
            'hull_indices_string': hull_indices_string
        })
    df_masks = pd.DataFrame(masks_data)  

    # merge df_masks and df_detections  
    df_detections = pd.merge(df_masks, df_boxes, how="outer", left_index=True, right_index=True)
    
    # clean database
    df_detections['image_id'] = image_id
    df_detections.rename(columns={'class_id_x': 'class_id'}, inplace=True)
    df_detections.drop(['class_id_y'], inplace=True, axis='columns')
    df_detections = df_detections.astype({'class_id': int, 'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int})

    return df_detections

# # Usage example:

# image_path = 'example_images/08hs-palms-03-zglw-superJumbo.webp'
# text_prompts = ["coconut palm tree"]
# results_gpu = run_sam3_semantic_predictor(input_image_path=image_path, text_prompts=text_prompts)
# results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
# # delete_results_from_gpu_memory() # Clear GPU memory after processing each image
# get_data_for_images_table(results_cpu)
# # fake_image_id = 999
# get_data_for_detections_table(results_cpu, image_id=fake_image_id)    


##############
# from efd2.py
##############

def clean_contour(contour: np.ndarray, sigma:float=1.0) -> np.ndarray:
    """Internal: Removes spikes and jitter."""
    x, y = contour[:, 0], contour[:, 1]
    x = median_filter(x, size=3, mode='wrap')
    y = median_filter(y, size=3, mode='wrap')
    x_smooth = gaussian_filter1d(x, sigma, mode='wrap')
    y_smooth = gaussian_filter1d(y, sigma, mode='wrap')
    return np.column_stack([x_smooth, y_smooth])


def calculate_efd(contour:np.ndarray, harmonics:int=20) -> np.ndarray:
    """Internal: Standard Kuhl and Giardina EFD math."""

    # Ensure the contour is closed
    if not np.allclose(contour[0], contour[-1]):
        ic('closing contour')
        contour = np.vstack([contour, contour[0]])

    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy**2).sum(axis=1))
    ic(np.min(dt))
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]

    coeffs = np.zeros((harmonics, 4))
    for n in range(1, harmonics + 1):
        term = 2 * np.pi * n / T
        factor = T / (2 * (n * np.pi)**2)

        an = factor * np.sum((dxy[:, 0] / dt) * (np.cos(term * t[1:]) - np.cos(term * t[:-1])))
        bn = factor * np.sum((dxy[:, 0] / dt) * (np.sin(term * t[1:]) - np.sin(term * t[:-1])))
        cn = factor * np.sum((dxy[:, 1] / dt) * (np.cos(term * t[1:]) - np.cos(term * t[:-1])))
        dn = factor * np.sum((dxy[:, 1] / dt) * (np.sin(term * t[1:]) - np.sin(term * t[:-1])))
         
        coeffs[n-1] = [an, bn, cn, dn]
    return coeffs


def get_feature_vector(contour:np.ndarray) -> np.ndarray:
    """Processes a polygon and returns a normalized 1D feature vector."""
    # cleaned = clean_contour(contour)
    # coeffs = calculate_efd(cleaned)
    coeffs = calculate_efd(contour, 50)
    return normalize(coeffs)


def normalize(coeffs:np.ndarray) -> np.ndarray:
    """Internal: Rotation, size, and starting-point invariance."""
    a1, b1, c1, d1 = coeffs[0]
    theta_1 = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), (a1**2 + c1**2 - b1**2 - d1**2))

    a1_p = a1 * np.cos(theta_1) + b1 * np.sin(theta_1)
    c1_p = c1 * np.cos(theta_1) + d1 * np.sin(theta_1)
    psi_1 = np.arctan2(c1_p, a1_p)
    E = np.sqrt(a1_p**2 + c1_p**2)

    norm_v = []
    for n in range(len(coeffs)):
        an, bn, cn, dn = coeffs[n]
        cos_nt = np.cos((n + 1) * theta_1)
        sin_nt = np.sin((n + 1) * theta_1)
        cp, sp = np.cos(psi_1), np.sin(psi_1)

        # Standard matrix transformations for normalization
        an_n = (1/E) * (cp * (an*cos_nt + bn*sin_nt) + sp * (cn*cos_nt + dn*sin_nt))
        bn_n = (1/E) * (cp * (-an*sin_nt + bn*cos_nt) + sp * (-cn*sin_nt + dn*cos_nt))
        cn_n = (1/E) * (-sp * (an*cos_nt + bn*sin_nt) + cp * (cn*cos_nt + dn*sin_nt))
        dn_n = (1/E) * (-sp * (-an*sin_nt + bn*cos_nt) + cp * (-cn*sin_nt + dn*cos_nt))
        norm_v.extend([an_n, bn_n, cn_n, dn_n])
    return np.array(norm_v)


def reconstruct(feature_vector:np.ndarray, num_points:int=200) -> tuple[np.ndarray, np.ndarray] :
    """Reconstructs (x, y) coordinates from a feature vector."""
    coeffs = feature_vector.reshape(-1, 4)
    t = np.linspace(0, 2 * np.pi, num_points)
    x, y = np.zeros(num_points), np.zeros(num_points)
    for n, (a, b, c, d) in enumerate(coeffs):
        h = n + 1
        x += a * np.cos(h * t) + b * np.sin(h * t)
        y += c * np.cos(h * t) + d * np.sin(h * t)
    return x, y

# Usage example

# # Initialize your analyzer
# analyzer = ShapeDescriptor(harmonics=15, smoothing_sigma=1.2)

# # Process two different shapes
# feat1 = analyzer.get_feature_vector(polygon_data_1)
# feat2 = analyzer.get_feature_vector(polygon_data_2)

# # Compare them
# dist = euclidean(feat1, feat2)
# print(f"Morphological distance: {dist:.4f}")

# # Visualize the normalized reconstruction
# x, y = analyzer.reconstruct(feat1)
# plt.plot(x, y)
# plt.axis('equal')
# plt.show()


def visualize_harmonics(contour: np.ndarray, harmonic_list: list[int]=[1, 3, 10, 50]) -> None:
    """Plots original vs reconstructed shapes for various harmonics."""
    plt.figure(figsize=(15, 5))

    # Plot Original
    plt.subplot(1, len(harmonic_list) + 1, 1)
    plt.plot(contour[:, 0], contour[:, 1], 'k--', alpha=0.5)
    plt.title("Original Boundary")
    plt.axis('equal')

    for i, n in enumerate(harmonic_list):
        coeffs = calculate_efd(contour, harmonics=n)
        # We don't use normalized coeffs for reconstruction overlay 
        # so they sit on top of the original shape
        rx, ry = reconstruct(coeffs)

        plt.subplot(1, len(harmonic_list) + 1, i + 2)
        plt.plot(contour[:, 0], contour[:, 1], 'k--', alpha=0.2)
        plt.plot(rx, ry, 'r')
        plt.title(f"Harmonics: {n}")
        plt.axis('equal')

    plt.tight_layout()
    plt.show()


def create_db(db_path: str) -> None:
    """ Creates an SQLite3 database"""

    # schema SQL script as a string
    sql_script = textwrap.dedent("""\
        --- 2026-03-31 17:30

        CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY,
        image_path TEXT UNIQUE,
        image_width INTEGER,
        image_height INTEGER,
        timestamp TEXT,
        latitude REAL,
        longitude REAL
        );

        CREATE TABLE IF NOT EXISTS detections (
        detection_id INTEGER PRIMARY KEY,
        image_id INTEGER,
        class_id INTEGER,
        poly_wkt TEXT,
        hull_indices_string TEXT,
        poly_wkt_c TEXT,
        x_min INTEGER,
        y_min INTEGER,
        x_max INTEGER,
        y_max INTEGER,
        confidence REAL,
        is_accepted INTEGER NOT NULL DEFAULT 0,
        is_healthy INTEGER NOT NULL DEFAULT 0,
        is_damaged_with_vcuts INTEGER NOT NULL DEFAULT 0,
        is_damaged_without_vcuts INTEGER NOT NULL DEFAULT 0,
        is_dead INTEGER NOT NULL DEFAULT 0,
        is_rejected INTEGER NOT NULL DEFAULT 0,
        is_crowded INTEGER NOT NULL DEFAULT 0,
        is_occluded INTEGER NOT NULL DEFAULT 0,
        has_other_problem INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE 
        );

        CREATE TABLE IF NOT EXISTS vcuts (
        vcut_id INTEGER PRIMARY KEY,
        detection_id INTEGER,
        start_x INTEGER,
        start_y INTEGER,
        far_x INTEGER,
        far_y INTEGER,        
        end_x INTEGER,
        end_y INTEGER,
        depth REAL,
        degrees REAL,
        FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
        );
        """)
    
    # Optional: Remove the database file if it already exists for a clean run
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        # Establish a connection (creates the DB file if it doesn't exist)
        conn = sqlite3.connect(db_path)
        print(f"Database {db_path} created/opened.")

        # Create a cursor object
        cursor = conn.cursor()

        # Execute the entire SQL script from the string
        cursor.executescript(sql_script)
        print("SQL script executed successfully.")

        # Commit the changes
        conn.commit()
        print("Changes committed.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback() # Roll back changes if an error occurs

    finally:
        # Close the connection
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

# Example usage:
# create_db("example.db")


def wkt2contour(wkt_str):
    """ Converts WKT polygon string to a contour array suitable for OpenCV functions. """
    return np.array(
        list(wkt_loads(wkt_str).exterior.coords), dtype=np.int32).reshape(-1, 1, 2)


def np_int_array_to_string(arr):
    """ Converts a numpy array of integers to a space-separated string. """
    arr_string = np.array2string(arr).replace('\n', '').replace('[', '').replace(']', '')
    return ' '.join(arr_string.split())


def string_to_np_int_array(s):
    """ Converts a space-separated string of integers back to a numpy array. """
    return np.fromstring(s, dtype=np.int32, sep=' ').reshape(-1, 1)


def get_data_for_vcuts_table(db_path, image_id:int)->pd.DataFrame:
    """ 
    Processes data for a single image. 
    Returns a pandas dataframe containing data to be added to the `vcuts` table.
    
    Presently, all data are sourced from the database. Undoubtedly, this is inefficient. 
    In the future, I will refactor to source data from the results_cpu variable that is 
    used for populating the `images` and `detections` tables.
    """
    # get input data from the images and detections tables
    sql = """ 
    SELECT image_path, detection_id, poly_wkt, hull_indices_string
    FROM images, detections
    WHERE images.image_id = detections.image_id
    """
    df_input = pd.read_sql(sql, sqlite3.connect(db_path))
    ic(df_input)

    data_list = [] # list of dicts to be converted to dataframe for insertion into vcuts table
    for i, r in df_input.iterrows():
        tree_contour = wkt2contour(r.poly_wkt)
        hull_indices = string_to_np_int_array(r.hull_indices_string)
        try:
            defects = cv2.convexityDefects(tree_contour, hull_indices)
        except Exception as e:
            print(f"Error occurred while calculating convexity defects for row {i}: {e}")
            continue
        # ic(defects)
        # ic(defects.shape)
        
        # process defects
        for j, defect in enumerate(defects):
            
            start_idx, end_idx, far_idx, depth = defect[0]
            start_point = tree_contour[start_idx][0]
            end_point = tree_contour[end_idx][0]
            far_point = tree_contour[far_idx][0]
            
            # calculate length of all sides of triangle in pixels
            a = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            b = math.sqrt((far_point[0] - start_point[0])**2 + (far_point[1] - start_point[1])**2)
            c = math.sqrt((end_point[0] - far_point[0])**2 + (end_point[1] - far_point[1])**2)
            
            # calculate depth of the convexity defect using Heron's formula in pixels
            # equals distance between far_point point and base of the triangle (the `a` side) 
            s = (a+b+c)/2
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            depth = (2*area)/a # depth is the height of the triangle, which is the distance from the far point to the line formed by the start and end points
            
            # calculate the angle of the defect in degrees using the cosine rule
            angle = math.degrees(math.acos((b**2 + c**2 - a**2)/(2*b*c)))
            # ic(i, d, angle, start, far, end)

            data_dict = {
                'detection_id': r.detection_id,
                'start_x': start_point[0],
                'start_y': start_point[1],
                'far_x': far_point[0],
                'far_y': far_point[1],
                'end_x': end_point[0],
                'end_y': end_point[1],
                'depth': depth,
                'degrees': angle
            }
            data_list.append(data_dict)
        df_output = pd.DataFrame(data_list)
    return df_output

# # Usage example:
# df_vcuts = get_data_for_vcuts_table(db_path=db_path, image_id=0)
# df_vcuts.to_sql('vcuts', sqlite3.connect(db_path), if_exists='delete_rows', index=False)


def test_build_db_with_single_image():
    image_path = 'example_images/08hs-palms-03-zglw-superJumbo.webp'
    db_path = 'test.db'    
    
    os.remove(db_path) if os.path.exists(db_path) else None
    
    # create test database and populate images table and detections table
    create_db(db_path)

    # run the SAM3 semantic predictor on a test image and get results on CPU for further processing
    results_gpu = run_sam3_semantic_predictor(
        input_image_path=image_path, 
        text_prompts=["coconut palm tree"]
    )
    results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU

    # Free up GPU memory in preparation for detecting objects in the next image
    # This is a work-around to prevent out-of-memory errors from the GPU
    # I move all results for further processing and use the GPU only for object detection.
    ic('copying results_gpu to results_cpu')
    results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
    ic('deleting results_gpu from GPU and clearing caches')       
    del results_gpu 
    gc.collect() 
    torch.cuda.empty_cache() # Clears unoccupied cached memory
    
    # populate 'images' table and 'detections' table with data from the test image
    df_images = get_data_for_images_table(results_cpu=results_cpu, image_path=image_path)
    df_images.to_sql('images', sqlite3.connect(db_path), if_exists='append', index=False)
    df_detections = get_data_for_detections_table(results_cpu=results_cpu, image_id=1)
    df_detections.to_sql('detections', sqlite3.connect(db_path), if_exists='append', index=False)
    df_vcuts = get_data_for_vcuts_table(db_path=db_path, image_id=1)
    df_vcuts.to_sql('vcuts', sqlite3.connect(db_path), if_exists='append', index=False)


# # MAIN
# test_build_db_with_single_image(
#     image_path="example_images/08hs-palms-03-zglw-superJumbo.webp",
#     db_path='test.db'
# )