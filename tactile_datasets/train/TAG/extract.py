""" The extract_frame.py is part of the data preprocessing of
our dataset that aims to transfer raw videos for both vision and
touch into static frames to support further applications. Specifically,
Two folders "video_frames" and "gelsight_frame" will be generated.
To run this code, please change the dir in line 57 to the path of our dataset.
"""

import os
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def getfiles(dir):
    filenames = os.listdir(dir)
    return filenames

def merge(dir):
    video1_dir = str(dir) + '/' + 'video.mp4'
    video2_dir = str(dir) + '/' + 'gelsight.mp4'

    cap1 = cv2.VideoCapture(video1_dir)
    frame_number1 = int(cap1.get(7))

    cap2 = cv2.VideoCapture(video2_dir)
    frame_number2 = int(cap2.get(7))
    
    frame_number1 = min(frame_number1, frame_number2)

    # Skip if already processed
    if os.path.exists(str(dir) + '/video_frame'):
        if len(os.listdir(str(dir) + '/video_frame')) == frame_number1:
            return
        else:
            print(str(dir))

    # Create directories if not exist
    os.makedirs(str(dir) + '/video_frame', exist_ok=True)
    os.makedirs(str(dir) + '/gelsight_frame', exist_ok=True)

    for i in range(frame_number1):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        cv2.imwrite(str(dir) + '/video_frame/' + str(i).rjust(10,'0') + '.jpg', frame1)
        cv2.imwrite(str(dir) + '/gelsight_frame/' + str(i).rjust(10,'0') + '.jpg', frame2)

    cap1.release()
    cap2.release()

def process_sub_dir(sub_dir):
    try:
        # print(f"Processing {sub_dir} started!")
        merge(sub_dir)
        # print(f"Processing {sub_dir} finished!")
        return True
    except Exception as e:
        print(f"Error processing {sub_dir}: {str(e)}")
        return False

def main():
    dir = Path('dataset/')  # Path to dataset
    files = getfiles(dir)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    
    # Create thread pool with optimal workers (adjust based on your system)
    max_workers = os.cpu_count() * 2  # Example: 2 threads per core
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sub_dir, dir / file): file for file in files}
        
        # Setup progress bar
        progress_bar = tqdm(total=len(futures), desc="Processing videos")
        
        # Handle completion and errors
        for future in as_completed(futures):
            sub_dir = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"\nError in {sub_dir}: {str(e)}")
            finally:
                progress_bar.update(1)
        progress_bar.close()

if __name__ == '__main__':
    main()