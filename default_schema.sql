-- default_schema.sql
/*
Author: Aubrey Moore (aubreymoore2013@gmail.com)
Timestamp: 2026-04-13 08:10
Purpose: 
*/

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT UNIQUE,
    image_width INTEGER,
    image_height INTEGER,
    timestamp TEXT,
    latitude REAL,
    longitude REAL
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    class_id INTEGER,
    tree_wkt TEXT,
    crown_wkt TEXT,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    confidence REAL,
    FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE 
);

CREATE TABLE IF NOT EXISTS vcuts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    start_x INTEGER,
    start_y INTEGER,
    far_x INTEGER,
    far_y INTEGER,        
    end_x INTEGER,
    end_y INTEGER,
    depth REAL,
    degrees REAL,
    emptiness REAL,
    FOREIGN KEY (detection_id) REFERENCES detections (id) ON DELETE CASCADE
);
