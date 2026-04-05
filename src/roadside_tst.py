"""

Note: This code is no longers used and will be removed from the repo at some point.

Tests are being moved into a single source code file: "roadside.py".

Docstring for roadside_test.py

Aubrey Moore (aubreymoore2013@gmail.com)
Last modified: 2026-03-06.

This python code provides tests for automated detection of coconut rhinoceros beetle 
damage in digital images provided in roadside.py.
"""

import roadside as rs
import os
from icecream import ic
import gc
import torch
import numpy as np
import sqlite3


def test_efd():
    # Create a dummy "star" shape with noise to simulate a complicated boundary
    t_orig = np.linspace(0, 2*np.pi, 100)
    r = 10 + 3*np.sin(5*t_orig) + np.random.normal(0, 0.2, 100)
    x_orig = r * np.cos(t_orig)
    y_orig = r * np.sin(t_orig)
    contour = np.column_stack([x_orig, y_orig])
    rs.visualize_harmonics(contour, [1,10, 20, 30, 40, 50])
    
# test_efd()


# def test_all() -> None:
#     """ 
#     # Runs only if GPU is available.
#     """
#     root_dir = "/home/aubrey/Desktop/crbdd"
#     image_paths = ["example_images/20251129_152106.jpg", "example_images/08hs-palms-03-zglw-superJumbo.webp" ]
#     text_prompts=["coconut palm tree"]
#     db_path = "sam3_detections.sqlite3"

#     os.chdir(root_dir) # ensure we start in the correct directory
#     if os.path.exists(db_path):
#         os.remove(db_path) # Remove existing database to start fresh

#     if rs.check_gpu(): # Only run if GPU is available 
#         ic('scanning images')      
#         for image_path in image_paths:
#             results_gpu = rs.run_sam3_semantic_predictor(
#                 input_image_path=image_path,
#                 text_prompts=text_prompts
#             )
#             ic(len(results_gpu))
            
#             # Free up GPU memory in preparation for detecting objects in the next image
#             # This is a work-around to prevent out-of-memory errors from the GPU
#             # I move all results for further processing and use the GPU only for object detection.
#             ic('copying results_gpu to results_cpu')
#             results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
#             ic('deleting results_gpu from GPU and clearing caches')       
#             del results_gpu 
#             gc.collect() 
#             torch.cuda.empty_cache() # Clears unoccupied cached memory

#     ic('FINISHED')
    
    
def test_all2() -> None:
    
    image_paths = [
        "example_images/08hs-palms-03-zglw-superJumbo.webp",
        "example_images/20251129_152106.jpg"
    ]
    text_prompts=["coconut palm tree"]
    db_path = 'sam3_detections.sqlite3'
    
    assert rs.check_gpu(), 'ERROR: GPU is unavailable.'

    rs.create_db(db_path)
    con = sqlite3.connect(db_path)

    # scan images and populate database
    for image_path in image_paths:
        
        # skip image if it is already in the database
        if con.execute(f'SELECT COUNT(*) FROM images WHERE  image_path = "{image_path}"').fetchone()[0] > 0:
            print(f'WARNING: Image {image_path} is already in the database. Skipping to next image.')
            continue
            
        # Detect objects in image
        results_gpu = rs.run_sam3_semantic_predictor(input_image_path=image_path, text_prompts=text_prompts)
        
        # Free up GPU memory in preparation for detecting objects in the next image
        # This is a work-around to prevent out-of-memory errors from the GPU
        # I move all results for further processing and use the GPU only for object detection.
        ic('copying results_gpu to results_cpu')
        results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
        ic('deleting results_gpu from GPU and clearing caches')       
        del results_gpu 
        gc.collect() 
        torch.cuda.empty_cache() # Clears unoccupied cached memory
        
        # populate 'images' table
        image_width, image_height, timestamp, latitude, longitude = rs.get_data_for_images_table(results_cpu, image_path)
        sql = """
        INSERT INTO images (image_path, image_width, image_height, timestamp, latitude, longitude) 
        VALUES (?,?,?,?,?,?)
        RETURNING image_id
        """
        parameters = (image_path, image_width, image_height, timestamp, latitude, longitude,)
        
        # sql = 'INSERT INTO images (image_path) VALUES (?) RETURNING image_id'
        try:
            image_id = con.execute(sql, parameters).fetchone()[0] # THE COMMA IN THE PARAMETERS TUPLE IS IMPORTANT
        except sqlite3.IntegrityError as e:
            print(f'ERROR: Image {image_path} already exists in {db_path}')
            raise e    
        con.commit()
        ic(image_id)
        
        # populate 'detections' table
        # df_detections = rs.get_data_for_detections_table(results_cpu, image_id)
        df_detections = rs.get_data_for_detections_table(results_cpu, image_id)
        for i, r in df_detections.iterrows():
                # populate 'detections' table
            sql = ''' 
            INSERT INTO detections
                (image_id, class_id, poly_wkt, poly_wkt_c, x_min, y_min, x_max, y_max, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            '''
            parameters = (image_id, 0, r['poly_wkt'], r['poly_wkt_c'], r['x_min'], r['y_min'], r['x_max'], r['y_max'], r['confidence']) 
            con.execute(sql, parameters)
            con.commit()
            
        # scan the 'detections' table for vcuts and populate 'vcuts' table  
        df_vcuts = rs.get_data_for_vcuts_table(db_path=db_path, image_id=0)
        df_vcuts.to_sql('vcuts', sqlite3.connect(db_path), if_exists='append', index=False)
        
    con.close()   
    print('FINISHED')
