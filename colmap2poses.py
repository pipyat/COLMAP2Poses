import numpy as np
import os

# Camera Extrinsics:

file_path = '/content/images.txt'
# Read the file and extract lines with the desired file type
file_type = '.jpg'
file_type_lines = []
with open(file_path, 'r') as file:
    for line in file:
        if file_type in line:
            file_type_lines.append(line.strip())

def get_poses(colmap_output):
    QW = float(colmap_output[1])
    QX = float(colmap_output[2])
    QY = float(colmap_output[3])
    QZ = float(colmap_output[4])
    TX = float(colmap_output[5])
    TY = float(colmap_output[6])
    TZ = float(colmap_output[7])

    # Normalise the quaternion components
    magnitude = (QW**2 + QX**2 + QY**2 + QZ**2)**0.5
    QW /= magnitude
    QX /= magnitude
    QY /= magnitude
    QZ /= magnitude

    # Calculate the rotation matrix
    R = np.array([
        [1 - 2*QY**2 - 2*QZ**2, 2*QX*QY - 2*QZ*QW, 2*QX*QZ + 2*QY*QW],
        [2*QX*QY + 2*QZ*QW, 1 - 2*QX**2 - 2*QZ**2, 2*QY*QZ - 2*QX*QW],
        [2*QX*QZ - 2*QY*QW, 2*QY*QZ + 2*QX*QW, 1 - 2*QX**2 - 2*QY**2]
    ])
    R = R.reshape((3,3))

    # Translation vector
    t = np.array([TX, TY, TZ])
    t = t.reshape((3,1))
    pose = np.column_stack((R,t,))
    pose = np.row_stack((pose,[0,0,0,1]))
    cam2world = np.linalg.inv(pose)

    return cam2world

poses = []
for line in file_type_lines:
    values = line.split()
    #print(values[0])
    pose = get_poses(values)
    poses.append(pose)

print("Must be close to zero:",np.max(np.linalg.inv(poses[0][:3,:3])-np.transpose(poses[0][:3,:3]))) # Checking that we have orthogonal rotation matrices

np.save('poses.npy',poses)

image_order = [] # for matching poses to the correct images
for line in file_type_lines:
    values = line.split()
    print(values[-1])
    image_order.append(values[-1])
np.save('image_order.npy',image_order)

# Optionally rename images to match the order of poses
def rename_images(directory):
    # Rename files in sequence
    for index, filename in enumerate(image_order):
        old_path = os.path.join(directory, filename)
        new_name = f"image{index}"+file_type
        new_path = os.path.join(directory, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {new_name}")
        else:
            print(f"File not found: {filename}")

# Specify the directory containing the images
directory = "/content"  # Change this to your actual directory
rename_images(directory)
