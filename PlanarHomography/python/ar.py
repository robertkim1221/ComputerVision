import cv2
import numpy as np
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import loadVid

def edit_frames(input_video_file, cover, ar_source, opts):
    # Open the input video file for reading
    input = loadVid(input_video_file)
    source = loadVid(ar_source)
    source = source[:, 50:310, 180:460, :]
    length = min(len(input), len(source))
    
    print(np.shape(source))
    new_img_array = []
    

    for i in range(length):
        #input frame
        input_frame = input[i]
    
        # Find matches between two images
        matches, locs1, locs2 = matchPics(cover, input_frame, opts)
        # Locs are in (y,x) format
        y1 = locs1[:, 0]
        x1 = locs1[:, 1]
        y2 = locs2[:, 0]
        x2 = locs2[:, 1]

        locs_x1 = np.array([x1, y1])
        locs_x1 = np.transpose(locs_x1)
        locs_x2 = np.array([x2, y2])
        locs_x2 = np.transpose(locs_x2)
        bestH2to1, inliers = computeH_ransac(locs_x1[matches[:, 0]], locs_x2[matches[:, 1]], opts)

        # preprocess the source video
        source_frame = source[i] #get the current frame
        source_frame = cv2.resize(source_frame, (cover.shape[1], cover.shape[0]))

        # Composite image
        composite_frame = compositeH(bestH2to1, source_frame, input_frame)

        print("Frame {} of {} processed".format(i + 1, length))

        # Write the composite frame to the output array
        frame = composite_frame
        new_img_array.append(frame)

    return new_img_array


if __name__ == "__main__":
    input_video_file = 'data/book.mov'  # Replace with your input video file
    ar_source = 'data/ar_source.mov'
    cover = cv2.imread('data/cv_cover.jpg')
    
    opts = get_opts()
    
    img_array = edit_frames(input_video_file, cover, ar_source, opts)

    # Define the codec and create a VideoWriter object for the output video
    out = cv2.VideoWriter('result/pandapotter.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (img_array[0].shape[1], img_array[0].shape[0]))

    for i in range(len(img_array)):
        img_array[i] *= 255/img_array[i].max() 
        out.write(np.array(img_array[i], np.uint8))
    
    out.release()

    