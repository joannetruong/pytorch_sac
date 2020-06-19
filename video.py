import imageio
import os
import numpy as np
import sys
from PIL import Image
import tqdm

import utils

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.quality = 10
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def resize_img(self, img, resolution):
        mode="L"
        if img.shape[-1] == 3:
            mode="RGB"
        resize_frame = Image.fromarray(np.squeeze(img).astype(np.uint8), mode=mode)
        resize_frame = np.array(resize_frame.resize((resolution, resolution), Image.BILINEAR))
        resize_frame = np.reshape(resize_frame, (resolution, resolution, -1))
        resize_frame =  resize_frame.astype(np.uint8)
        return resize_frame

    def record_rgb(self, env):
        rgb_frame = env.get_rgb()
        return rgb_frame.astype(np.uint8)

    def record_depth(self, env):
        depth_frame = env.get_depth()
        depth_frame =  depth_frame.squeeze().astype(np.uint8)
        depth_frame = np.stack([depth_frame for _ in range(3)], axis=2)
        return depth_frame

    def record_map(self, env):
        map_frame = env.get_map()
#        map_frame =  map_frame.squeeze().astype(np.uint8)
#        map_frame = np.stack([map_frame for _ in range(3)], axis=2)
        return map_frame.astype(np.uint8)
        #self.map_frame = Image.fromarray((map_frame).astype(np.uint8), mode="L")

    def record(self, env, params):
        ego_frames = []
        if 'rgb' in params:
            rgb_frame = self.record_rgb(env)
            ego_frames.append(self.resize_img(rgb_frame, 256))
        if 'depth' in params:
            depth_frame = self.record_depth(env)
            ego_frames.append(self.resize_img(depth_frame, 256))
        ego_frames = np.concatenate(ego_frames, axis=1)
        frames = ego_frames
        if 'map' in params:
            map_frame = self.record_map(env)
            map_frame = self.resize_img(map_frame, 256)
            frames = np.concatenate((ego_frames, map_frame), axis=1)
        self.frames.append(frames)

    def save(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        print('len frames: ', len(self.frames))
        imageio.mimsave(path, self.frames, fps=self.fps)
