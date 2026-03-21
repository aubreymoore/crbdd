import roadside as rs
import os
from icecream import ic
import gc
import torch

import numpy as np


def test_efd():
    # Create a dummy "star" shape with noise to simulate a complicated boundary
    t_orig = np.linspace(0, 2*np.pi, 100)
    r = 10 + 3*np.sin(5*t_orig) + np.random.normal(0, 0.2, 100)
    x_orig = r * np.cos(t_orig)
    y_orig = r * np.sin(t_orig)
    contour = np.column_stack([x_orig, y_orig])
    rs.visualize_harmonics(contour, [1,10, 20, 30, 40, 50])
    
# test_efd()


def test_all() -> None:
    """ 
    # Runs only if GPU is available.
    """
    root_dir = "/home/aubrey/Desktop/sam3-2026-01-31"
    image_paths = ["20251129_152106.jpg", "08hs-palms-03-zglw-superJumbo.webp" ]
    text_prompts=["coconut palm tree"]
    db_path = "sam3_detections.sqlite3"

    os.chdir(root_dir) # ensure we start in the correct directory
    if os.path.exists(db_path):
        os.remove(db_path) # Remove existing database to start fresh

    if rs.check_gpu(): # Only run if GPU is available 
        ic('scanning images')      
        for image_path in image_paths:
            results_gpu = rs.run_sam3_semantic_predictor(
                input_image_path=image_path,
                text_prompts=text_prompts
            )
            ic(len(results_gpu))
            

            # Free up GPU memory in preparation for detecting objects in the next image
            # This is a work-around to prevent out-of-memory errors from the GPU
            # I move all results for further processing and use the GPU only for object detection.
            ic('copying results_gpu to results_cpu')
            results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
            ic('deleting results_gpu from GPU and clearing caches')       
            del results_gpu 
            gc.collect() 
            torch.cuda.empty_cache() # Clears unoccupied cached memory

            ic('building database') 
            rs.build_database(
                image_path=image_path,
                results=results_cpu,                             
                db_path=db_path
            ) 

    ic('FINISHED')
   

