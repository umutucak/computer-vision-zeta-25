import glob
import os

import cv2
import numpy as np
import pandas as pd
from random import randint
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


def project(q, r, K):
    """ Projecting points to image frame to draw axes """
    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)
    # transformation to camera frame
    pose_mat = np.hstack((Rotation.from_quat(q).as_matrix(), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)
    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]
    # projection to image plane
    points_image_plane = K.dot(points_camera_frame)
    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y


def visualize(img, q, r, K, ax=None):
    """ Visualizing image, with ground truth pose with axes projected to training image. """
    if ax is None:
        ax = plt.gca()

    ax.imshow(img)
    xa, ya = project(q, r, K)
    scale = 150
    c = np.array([[xa[0]],
                  [ya[0]]
                  ])
    p = np.array([[xa[1], xa[2], xa[3]],
                  [ya[1], ya[2], ya[3]]
                  ])
    v = p - c
    v = scale * v / np.linalg.norm(v)
    ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color='r')
    ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color='g')
    ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color='b')
    return


if __name__ == '__main__':
    home = os.path.expanduser("~")
    root_dir = 'datasets/spark'
    dataset_dir = os.path.join(home, root_dir)

    dir_list = glob.glob(os.path.join(dataset_dir, "images", "*", ""), recursive=True)
    # visualise image from first directory
    traj_dir = os.path.basename(os.path.dirname(dir_list[randint(0, len(dir_list) - 1)]))
    processed_file_path = os.path.join(dataset_dir, "train.csv")

    # Read the string from the txt file
    with open(os.path.join(dataset_dir, 'K.txt'), 'r') as file:
        array_string = file.read()

    # Convert the string representation of the array to a numpy array
    K = np.array(eval(array_string))

    data = pd.read_csv(processed_file_path)
    image_path = os.path.join(dataset_dir, 'images', traj_dir)
    files = [os.path.basename(x) for x in glob.glob(os.path.join(image_path, '*.jpg'))]
    rows = 3
    cols = 3
    k = 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i in range(rows):
        for j in range(cols):
            k = randint(0, 300)
            image_id = files[k]
            print(image_id)
            img_path = os.path.join(image_path, files[k])
            im_read = cv2.imread(img_path)
            image = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
            i_data = data.loc[data['filename'] == image_id]
            Tx = i_data['Tx'].values.squeeze()
            Ty = i_data['Ty'].values.squeeze()
            Tz = i_data['Tz'].values.squeeze()
            # Qx, Qy, Qz, Qw
            Qx = i_data['Qx'].values.squeeze()
            Qy = i_data['Qy'].values.squeeze()
            Qz = i_data['Qz'].values.squeeze()
            Qw = i_data['Qw'].values.squeeze()

            r = np.array([Tx, Ty, Tz])
            q = np.array([Qx, Qy, Qz, Qw])
            print(r, q)
            visualize(image, q, r, K, ax=axes[i][j])
            axes[i][j].axis('off')
            k += 1
    fig.tight_layout()
    plt.show()
