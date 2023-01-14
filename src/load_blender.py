# implemented refer to InfoNeRF/dataset/load_blender.py
import os, json, imageio
import numpy as np
import jittor as jt

def pose_spherical(_theta, _phi, radius):
    phi = _phi / 180. * np.pi
    theta = _theta / 180. * np.pi
    translate = jt.array(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1]]).astype(np.float32))
    rotate_phi = jt.array(np.array([
        [1, 0,           0,            0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi),  0],
        [0, 0,           0,            1]]).astype(np.float32))
    rotate_theta = jt.array(np.array([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0,             1, 0,              0],
        [np.sin(theta), 0, np.cos(theta),  0],
        [0,             0, 0,              1]]).astype(np.float32))
    c2w = translate
    c2w = rotate_phi @ c2w
    c2w = rotate_theta @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(dataset_path):
    imgs_list = []
    poses_list = []
    counts = [0]
    for split in ['train', 'val', 'test']:
        imgs = []
        poses = []
        # print(os.path.join(dataset_path, 'transforms_{}.json'.format(split))
        with open(os.path.join(dataset_path, 'transforms_{}.json'.format(split)), 'r') as f:
            data = json.load(f)
            if split == 'train':
                skip = 1
            else:
                skip = 8 # load 1/8 images from test/val sets
            for frame in data['frames'][::skip]:
                imgs.append(imageio.imread(os.path.join(dataset_path, frame['file_path'] + '.png')))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32)
            imgs_list.append(imgs)
            poses = np.array(poses).astype(np.float32)
            poses_list.append(poses)

            counts.append(counts[-1] + len(imgs)) # for splitting train/val/test
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(imgs_list, 0)
    poses = np.concatenate(poses_list, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(data['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = jt.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
        
    return imgs, poses, render_poses, [H, W, focal], i_split

