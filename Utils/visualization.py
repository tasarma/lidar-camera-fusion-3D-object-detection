import os 
import sys
import math

import cv2
import numpy as np
import open3d as o3d
# from mayavi import mlab
import matplotlib.pyplot as plt

from . import init_path
from DataProcess import kitti_utils, transformation
import Config.kitti_config as cnf


def show_image_with_boxes(img, objects, calib, show3d=False):
    ''' Show image with 2D bounding boxes '''
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.cls_type == 'DontCare': continue
        cv2.rectangle(img, (int(obj.xmin),int(obj.ymin)),
           (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P2)
        if box3d_pts_2d is not None:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, cnf.colors[obj.cls_id])
    if show3d:
        cv2.imshow("img", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img2
    else:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img


def display_lidar(cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])


def draw_projected_box3d(image, qs, color=(255, 0, 255), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def draw_gt_boxes3d(gt_boxes3d, 
                    fig, color=(1, 1, 1), 
                    line_width=2, 
                    draw_text=True, 
                    text_scale=(1, 1, 1),
                    color_list=None
    ):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.show()
    return fig

def show_lidar_with_boxes(lidar, labels, calib):
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))

    mlab.points3d(lidar[:, 0], lidar[:, 1], lidar[:, 2], mode="point", colormap="spectral", figure=fig)
    
    # Plot the bounding boxes
    for obj in labels:
        if obj.cls_type == 'DontCare':
            continue

        box3d_pts_2d, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P2)
        corners_3d_in_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print(corners_3d_in_velo)
        draw_gt_boxes3d([corners_3d_in_velo], fig=fig, color=(0, 1, 1), line_width=2, draw_text=True)

    mlab.view(azimuth=230, distance=50)
    # mlab.savefig(filename='examples/kitti_3dbox_to_cloud.png')
    mlab.show()


def render_pcl(pc, box =  [], name = "default"):
   print(pc)
   fig=plt.figure(name)
   ax = fig.gca(projection='3d')
    
   X = pc[0]
   Y = pc[1]
   Z = pc[2]
   ax.scatter(X, Y, Z, s=1)
   
   # pcl characteristics 
   print("X maximum is :" + str(X.max()))
   print("Y maximum is :" + str(Y.max()))
   print("Z maximum is :" + str(Z.max()))

   max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
   Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
   Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
   Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

   i=0
   for xb, yb, zb in zip(Xb, Yb, Zb):
      i = i+1
      ax.plot([xb], [yb], [zb], 'b')
   if len(box)!=0:
       x = box[0] 
       y = box[1]
       z = box[2]
       ax.plot(x, y, z, color = 'r')
   return ax

def visualize_result(anchor_point, offset, gt_boxes):
   for i in range(0,4):
      final_pred = np.zeros((8,3))
      final_pred = offset[i] + anchor_point[i, None]
      render_pcl(final_pred.T, name = str(i))
      render_pcl(gt_boxes[i].T, name = str(i))
   plt.show()