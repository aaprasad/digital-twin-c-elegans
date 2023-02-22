import cv2
import os


# Function to extract frames
def FrameCapture(folder, video_name):
    # Path to video file
    vidObj = cv2.VideoCapture(os.path.join(folder, video_name))
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    os.makedirs(os.path.join(folder, 'frames'), exist_ok=True)
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # Saves the frames with frame-count
        cv2.imwrite(os.path.join(folder, 'frames', "frame{}.jpg".format(count)), image)
        count += 1


# Driver Code
if __name__ == '__main__':
    # Calling the function
    # folder, video_name = 'video/swimmer_256', 'openaigym.video.0.80647.video000000.mp4'
    folder, video_name = 'video/swimmer_turn_160', 'openaigym.video.0.84937.video000000.mp4'
    FrameCapture(folder, video_name)
