import logging
import json
from time import time
from time import sleep
from types import SimpleNamespace

import numpy as np
import open3d as o3d

import a0
from polygrasp import serdes

from transforms3d.euler import euler2mat
import yaml
import xml.etree.cElementTree as ET

log = logging.getLogger(__name__)
topic = "cams/rgbd"
# topic = "color_image_compressed"

import cv2
import time

class CameraSubscriber:
    def __init__(self, intrinsics_file, extrinsics_file):
        with open(intrinsics_file, "r") as f:
            intrinsics_json = json.load(f)
            self.intrinsics = [SimpleNamespace(**d) for d in intrinsics_json]

        with open(extrinsics_file, "r") as f:
            self.extrinsics = json.load(f)
        print(len(self.intrinsics))
        print(len(self.extrinsics))
        assert len(self.intrinsics) == len(self.extrinsics)
        self.n_cams = len(self.intrinsics)
        self.done = False
        self.rgbds = None
        self.sub = None
        # self.sub = a0.RemoteSubscriber("172.16.0.3", topic, self.cback, a0.INIT_AWAIT_NEW, a0.ITER_NEXT)
        self.sub = a0.RemoteSubscriber("172.16.0.3", topic, self.cback, a0.INIT_AWAIT_NEW, a0.ITER_NEWEST)
        # self.sub = a0.RemoteSubscriber("172.16.0.3", topic, self.cback, a0.INIT_MOST_RECENT, a0.ITER_NEWEST)
        # self.sub = a0.SubscriberSync(topic, a0.INIT_MOST_RECENT, a0.ITER_NEWEST)
        self.recent_rgbd = None

    def cback(self, pkt):
        # self.done = True
        self.recent_rgbd = serdes.bytes_to_np(pkt.payload)
        if ((self.recent_rgbd is not None) and len(self.recent_rgbd)>0):
            self.done = True
        # print(f"Got {pkt.payload}")
        # print("Got message")

    def get_rgbd(self):
        if self.sub.can_read():
            self.recent_rgbd = serdes.bytes_to_np(self.sub.read().payload)
        return self.recent_rgbd


class PointCloudSubscriber(CameraSubscriber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = self.intrinsics[0].width
        self.height = self.intrinsics[0].height

        # Convert to open3d intrinsics
        self.o3_intrinsics = [
            o3d.camera.PinholeCameraIntrinsic(
                width=intrinsic.width,
                height=intrinsic.height,
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.ppx,
                cy=intrinsic.ppy,
            )
            for intrinsic in self.intrinsics
        ]

        # Convert to numpy homogeneous transforms
        self.extrinsic_transforms = np.empty([self.n_cams, 4, 4])
        for i, calibration in enumerate(self.extrinsics):
            self.extrinsic_transforms[i] = np.eye(4)
            self.extrinsic_transforms[i, :3, :3] = calibration["camera_base_ori"]
            self.extrinsic_transforms[i, :3, 3] = calibration["camera_base_pos"]

    def get_pcd_i(self, rgbd: np.ndarray, cam_i: int, mask: np.ndarray = None):
        if mask is None:
            mask = np.ones([self.height, self.width])

        intrinsic = self.o3_intrinsics[cam_i]
        transform = self.extrinsic_transforms[cam_i]

        # The specific casting here seems to be very important, even though
        # rgbd should already be in np.uint16 type...
        img = (rgbd[:, :, :3] * mask[:, :, None]).astype(np.uint8)
        depth = (rgbd[:, :, 3] * mask).astype(np.uint16)

        o3d_img = o3d.cuda.pybind.geometry.Image(img)
        o3d_depth = o3d.cuda.pybind.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_img,
            o3d_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform(transform)

        return pcd

    def get_pcd(
        self, rgbds: np.ndarray, masks: np.ndarray = None
    ) -> o3d.geometry.PointCloud:
        if masks is None:
            masks = np.ones([self.n_cams, self.height, self.width])
        pcds = [self.get_pcd_i(rgbds[i], i, masks[i]) for i in range(len(rgbds))]
        result = pcds[0]
        for pcd in pcds[1:]:
            result += pcd
        return result

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

if __name__ == "__main__":   
    pcs = PointCloudSubscriber("conf/intrinsics.json", "conf/extrinsics.json")
    while(pcs.done != True):
        sleep(0.1)
        print("Waiting")
    print("Done waiting")
    rgbds = pcs.recent_rgbd
    # pcs.sub = None
    rgbd = rgbds[0]
    frame = rgbd[:,:,0:3]
    frame = cv2.convertScaleAbs(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    pcs.done = False
    saved_hsv = np.load('hsv_value.npy')
    # # Initializing the webcam feed.
    # cap = cv2.VideoCapture(4)

    # Create a window named trackbars.
    cv2.namedWindow("Trackbars")

    # Now create 6 trackbars that will control the lower and upper range of 
    # H,S and V channels. The Arguments are like this: Name of trackbar, 
    # window name, range,callback function. For Hue the range is 0-179 and
    # for S,V its 0-255.
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    
    while True:
        while(pcs.done != True):
            sleep(0.1)
            # print("Waiting")
        rgbds = pcs.recent_rgbd
        # pcs.sub = None
        rgbd = rgbds[0]
        frame = rgbd[:,:,0:3]
        frame = cv2.convertScaleAbs(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Start reading the webcam feed frame by frame.
        # ret, frame = cap.read()
        # if not ret:
        #     break
        # Flip the frame horizontally (Not required)
        
        # frame = cv2.flip( frame, 1 ) 
        
        # Convert the BGR image to HSV image.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the new values of the trackbar in real time as the user changes 
        # them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
        pcs.done = False
        lower_range = saved_hsv[0]
        upper_range = saved_hsv[1]
        
        # Filter the image and get the binary mask, where white represents 
        # your target color
        mask = cv2.inRange(hsv, lower_range, upper_range)
    
        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Converting the binary mask to 3 channel image, this is just so 
        # we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((mask_3,frame,res))
        
        # Show this stacked frame at 40% of the size.
        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
        
        # If the user presses ESC then exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        # If the user presses `s` then print this array.
        if key == ord('s'):
            
            thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
            print(thearray)
            
            # Also save this array as penval.npy
            np.save('hsv_value',thearray)
            break
        
    # Release the camera & destroy the windows.    
    # cap.release()
    pcs.sub = None
    cv2.destroyAllWindows()