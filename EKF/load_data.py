from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2


mat_data = loadmat(r"D:\Spring 2024\Adv Nav\Codes\Nonlinear Kalman Filter\data\studentdata0.mat", simplify_cells=True)

data_list = mat_data['data']
time = mat_data['time']

# Initialize empty lists
img_list = []
id_array_list = []
p1_list = []
p2_list = []
p3_list = []
p4_list = []
timestamp_list = []
rpy_list = []
omg_list = []
acc_list = []

# Iterate through each data in the data_list
for data in data_list:
    # Append data from each trial to the respective lists
    img_list.append(data['img'])
    id_array_list.append(data['id'])
    p1_list.append(data['p1'])
    p2_list.append(data['p2'])
    p3_list.append(data['p3'])
    p4_list.append(data['p4'])
    timestamp_list.append(data['t'])
    rpy_list.append(data['rpy'])
    omg_list.append(data['drpy'])
    acc_list.append(data['acc'])


vicon_data = mat_data['vicon']

# Extract estimated positions from xhat
p_hat = [sublist[:3] for sublist in vicon_data]

# Extract x, y, z coordinates
x = vicon_data[0]
y = vicon_data[1]
z = vicon_data[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.plot(x, y, z, c='r', linewidth=2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# plt.show()