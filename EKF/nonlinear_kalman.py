import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KalmanFilter:
    def __init__(self, data_path):
        self.data_path = data_path
        # use self to store data
        self.img_list = []
        self.id_array_list = []
        self.p1_list = []
        self.p2_list = []
        self.p3_list = []
        self.p4_list = []
        self.timestamp_list = []
        self.rpy_list = []
        self.omg_list = []
        self.acc_list = []
        self.vicon_data = []
        self.time = []
        self.image_coordinates = {}
        self.pos = []
        self.rot = []

    def load_data(self):
        mat_data = loadmat(self.data_path, simplify_cells=True)

        data_list = mat_data['data']
        self.time = mat_data['time']

        # Iterate through each data in the data_list
        for index, data in enumerate(data_list):
            # Append data from each trial to the respective lists
            self.img_list.append(data['img'])
            self.id_array_list.append(data['id'])
            self.p1_list.append(data['p1'])
            self.p2_list.append(data['p2'])
            self.p3_list.append(data['p3'])
            self.p4_list.append(data['p4'])

            self.rpy_list.append(data['rpy'])
            self.omg_list.append(data['drpy'])
            self.acc_list.append(data['acc'])

            self.image_coordinates[index] = {}
            if isinstance(data['id'], int):

                self.image_coordinates[index][data['id']] = np.array([[data['p1'][0], data['p1'][1]],
                                                                    [data['p2'][0], data['p2'][1]],
                                                                    [data['p3'][0], data['p3'][1]],
                                                                    [data['p4'][0], data['p4'][1]]])
                self.timestamp_list.append(data['t'])
                continue

            for idx, tag in enumerate(data['id']):
                self.image_coordinates[index][tag] = np.array([[data['p1'][0][idx], data['p1'][1][idx]],
                                                                    [data['p2'][0][idx], data['p2'][1][idx]],
                                                                    [data['p3'][0][idx], data['p3'][1][idx]],
                                                                    [data['p4'][0][idx], data['p4'][1][idx]]])
            if data['id'].shape[0] > 1:
                self.timestamp_list.append(data['t'])


        self.vicon_data = mat_data['vicon']

        return data_list


    def world_coordinates(self):
        position = [
            [0, 12, 24, 36, 48, 60, 72, 84, 96],
            [1, 13, 25, 37, 49, 61, 73, 85, 97],
            [2, 14, 26, 38, 50, 62, 74, 86, 98],
            [3, 15, 27, 39, 51, 63, 75, 87, 99],
            [4, 16, 28, 40, 52, 64, 76, 88, 100],
            [5, 17, 29, 41, 53, 65, 77, 89, 101],
            [6, 18, 30, 42, 54, 66, 78, 90, 102],
            [7, 19, 31, 43, 55, 67, 79, 91, 103],
            [8, 20, 32, 44, 56, 68, 80, 92, 104],
            [9, 21, 33, 45, 57, 69, 81, 93, 105],
            [10, 22, 34, 46, 58, 70, 82, 94, 106],
            [11, 23, 35, 47, 59, 71, 83, 95, 107]
        ]
        tag_position = np.array(position)

        # Create dictionary to store the 4 corners of the AprilTag
        self.tag_corners = {}

        tag_size = 0.152  # in meters

        # Define the spacing between tags
        spacing_x = 0.152  # in meters
        spacing_y = 0.152  # in meters
        special_spacing_y = 0.178  # in meters

        # Iterate through each tag in the tag_position
        for i in range(tag_position.shape[0]):
            for j in range(tag_position.shape[1]):
                # Append the 4 corners of the AprilTag to the dictionary
                # Each tag is 0.152 m x 0.152 m in size
                # The 4 corners are in the following order: top left, top right, bottom right, bottom left
                # The corners are in the following order: x, y, z where z is always 0
                # the distance between the tags is 0.152 m in the x and y direction except for the space between columns 3 and 4, and 6 and 7, which is 0.178 m
                x = i * spacing_x * 2
                y = j * spacing_y * 2

                # Adjust y coordinate for special spacing
                if j + 1 >= 3:
                    y += (special_spacing_y - spacing_y)
                if j + 1 >= 6:
                    y += (special_spacing_y - spacing_y)
                self.tag_corners[tag_position[i, j]] = np.array([
                                                        [x + tag_size, y, 0],  # bottom left corner
                                                        [x + tag_size, y + tag_size, 0],  # bottom right corner
                                                        [x, y + tag_size, 0],  # top right corner
                                                        [x, y, 0] # top left corner
                                                    ])
        # print(f"Tag Corners {self.tag_corners}")

    def camera_params(self):
        self.camera_matrix = np.array([[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float32)

    def estimate_pose(self, tags, idx):

        image_points = []
        object_points = []

        for tag in tags:
            image_points.append(self.image_coordinates[idx][tag][0])
            image_points.append(self.image_coordinates[idx][tag][1])
            image_points.append(self.image_coordinates[idx][tag][2])
            image_points.append(self.image_coordinates[idx][tag][3])

            object_points.append(self.tag_corners[tag][0])
            object_points.append(self.tag_corners[tag][1])
            object_points.append(self.tag_corners[tag][2])
            object_points.append(self.tag_corners[tag][3])

        ret, rvec, tvec = cv2.solvePnP(np.array(object_points, dtype=np.float32), np.array(image_points, dtype=np.float32),
                                      self.camera_matrix, self.dist_coeffs)


        camera_position = np.array([-0.04, 0.0, -0.03])  # XYZ coordinates of the camera

        yaw = np.pi / 4 # Yaw angle of the camera

        # Create rotation matrix from yaw angle
        rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

        rotation_x = np.array([[1, 0, 0],
                        [0, np.cos(np.pi), -np.sin(np.pi)],
                        [0, np.sin(np.pi), np.cos(np.pi)]])
        camera_R = rotation_x @ rotation_z

        M_camera_drone = np.eye(4)
        M_camera_drone[:3, :3] = camera_R
        M_camera_drone[:3, 3] = camera_position

        R, _ = cv2.Rodrigues(rvec)

        M_camera_object = np.eye(4)
        M_camera_object[:3, :3] = R
        M_camera_object[:3, 3] = tvec.flatten()

        M_object_drone = np.linalg.inv(M_camera_object) @ M_camera_drone

        R_object_drone = self.rotation_to_euler(M_object_drone[:3, :3])
        t_object_drone = M_object_drone[:3, 3]

        self.pos.append([t_object_drone[0], t_object_drone[1], t_object_drone[2]])
        self.rot.append([R_object_drone[0], R_object_drone[1], R_object_drone[2]])

    def rotation_to_euler(self, R):
        """
        Convert a rotation matrix to Euler angles yaw, pitch, roll
        :param R:
        :return:
        """
        yaw = np.arctan(-R[0, 1] / R[1, 1])
        roll = np.arctan(R[2,1] * np.cos(yaw) / R[1,1])
        pitch = np.arctan(-R[2, 0] / R[2, 2])

        # yaw = np.arctan2(R[1, 0], R[0, 0])
        # pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        # roll = np.arctan2(R[2, 1], R[2, 2])

        return np.array([yaw, pitch, roll])

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(self.vicon_data[0], self.vicon_data[1], self.vicon_data[2])
        self.pos = np.array(self.pos)
        ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        # ax.plot(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], c='r', linewidth=2)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Plot to compare estimated angle with ground truth
        fig, ax = plt.subplots(3, 1)
        ax[0].set_ylim(-np.pi, np.pi)
        ax[0].plot(self.time, self.vicon_data[2], label='Yaw')
        ax[0].plot(np.array(self.timestamp_list), np.array([rot[0] for rot in self.rot]), label='Estimated Yaw')
        ax[0].set_title('Yaw')
        ax[0].legend()

        ax[1].set_ylim(-np.pi, np.pi)
        ax[1].plot(self.time, self.vicon_data[1], label='Pitch')
        ax[1].plot(np.array(self.timestamp_list), np.array([rot[2] for rot in self.rot]), label='Estimated Pitch')
        ax[1].set_title('Pitch')
        ax[1].legend()

        ax[2].set_ylim(-np.pi, np.pi)
        ax[2].plot(self.time, self.vicon_data[0], label='Roll')
        ax[2].plot(np.array(self.timestamp_list), np.array([rot[1] for rot in self.rot]), label='Estimated Roll')
        ax[2].set_title('Roll')
        ax[2].legend()

        plt.show()



if __name__ == "__main__":
    data_path = r"D:\Spring 2024\Adv Nav\Codes\Nonlinear Kalman Filter\data\studentdata0.mat"
    filter = KalmanFilter(data_path)
    data_list = filter.load_data()
    filter.world_coordinates()
    filter.camera_params()
    for idx, data in enumerate(data_list):
        if isinstance(data['id'], int):
            filter.estimate_pose([data['id']], idx)
            continue
        if data['id'].shape[0] > 1:
            filter.estimate_pose(data['id'], idx)
    filter.visualize()










