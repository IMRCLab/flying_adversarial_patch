#!/usr/bin/env python

import numpy as np
import rowan
from pathlib import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from functools import partial


from crazyflie_py import *
from crazyflie_py.uav_trajectory import Trajectory

from tf2_ros import TransformBroadcaster

import yaml


# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q

class Attack():
    attacker_id = 4
    victim_id = 18

    def __init__(self):
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.node = self.swarm.allcfs

        self.pose_a = None
        self.subscription_a = self.node.create_subscription(
                PoseStamped,
                f'cf{self.attacker_id}/pose',
                partial(self.pose_callback, "a"),
                10)
        
        self.pose_v = None
        self.subscription_v = self.node.create_subscription(
                PoseStamped,
                f'cf{self.victim_id}/pose',
                partial(self.pose_callback, "v"),
                10)
        
        self.cf_a = self.swarm.allcfs.crazyfliesById[self.attacker_id]
        self.cf_v = self.swarm.allcfs.crazyfliesById[self.victim_id]

        self.tfbr = TransformBroadcaster(self.node)

        
    def pose_callback(self, store_into, msg: PoseStamped):
        # store the latest pose
        if store_into == 'a':
            self.pose_a = msg
        else:
            self.pose_v = msg

    def _broadcast(self, name, frame, T):
        # helper function to broadcast poses to rviz
        t_base = TransformStamped()
        t_base.header.stamp = self.node.get_clock().now().to_msg()
        t_base.header.frame_id = frame
        t_base.child_frame_id = name

        qw, qx, qy, qz = rowan.from_matrix(T[:3, :3])
        x, y, z = T[:3, 3]


        t_base.transform.translation.x = x
        t_base.transform.translation.y = y
        t_base.transform.translation.z = z
        t_base.transform.rotation.x = qx
        t_base.transform.rotation.y = qy
        t_base.transform.rotation.z = qz
        t_base.transform.rotation.w = qw
        self.tfbr.sendTransform(t_base)

    def _get_T(self, msg):
        # calculate full 4x4 transformation matrix from position and quaternions
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]

        rot = rowan.to_matrix(quat)
        T = np.zeros((4,4))
        T[:3, :3] = rot
        T[:3, 3] = pos
        T[-1, -1] = 1
        return T

    # mimics the control law of frontnet (see app.c), to compute the setpoint, assuming a certain NN prediction
    # target_pos_nn is the NN prediction (target position in UAV frame)
    def _compute_frontnet_reaction(self, T_victim_world_c, target_pos_nn):
        # get the current 3D coordinate of the UAV
        p_D = T_victim_world_c[:3, 3]

        # translate the target pose in world frame
        q = rowan.from_matrix(T_victim_world_c[:3, :3])
        target_pos = p_D + rowan.rotate(q, target_pos_nn)

        # calculate the UAV's yaw angle
        target_drone_global = target_pos - p_D
        target_yaw = np.arctan2(target_drone_global[1], target_drone_global[0])

        # eq 6
        distance = 1.0
        def calcHeadingVec(radius, angle):
            return np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

        e_H_delta = calcHeadingVec(1.0*distance, target_yaw-np.pi)

        p_D_prime = target_pos + e_H_delta

        return np.array([p_D_prime[0], p_D_prime[1], 1.0]), target_yaw

    def compute_attacker_pose(self, pos_v_desired, targets, A, positions):

        # broadcast poses to rviz
        T_victim_world_c = self._get_T(self.pose_v)

        self._broadcast('v', 'world', T_victim_world_c)

        T_attacker_world = self._get_T(self.pose_a)

        self._broadcast('a', 'world', T_attacker_world)


        T_victim_world_d = np.eye(4)
        T_victim_world_d[:3, 3] = pos_v_desired

        self._broadcast('vd', 'world', T_victim_world_d)

        # decide which patch + position would move the victim closer to the desired trajectory
        best_attack = None
        best_error = np.inf
        for target, a, pos in zip(targets, A, positions):
            pos_v_effect, theta_v_effect = self._compute_frontnet_reaction(T_victim_world_c, target) # mimic the output of frontnet
            error = np.linalg.norm(pos_v_desired - pos_v_effect)
            if error < best_error:
                best_error = error
                best_attack = target, a, pos


        # apply this patch relative to the current position
        T_patch_in_victim = np.eye(4)
        T_patch_in_victim[:3, 3] = best_attack[2]  # pos in best_attack is position of the patch relative to victim
        T_patch_in_victim[2, 3] += 0.093   # center of the patch is 9.3 cm underneath the center of Crazyflie, therefore z needs to be shifted

        T_attacker_in_world = T_victim_world_c @ T_patch_in_victim # transform attacker pose into world frame

        self._broadcast('ad', 'world', T_attacker_in_world)


        # return desired pos and yaw for the attacker
        roll, pitch, yaw = rowan.to_euler(rowan.from_matrix(T_attacker_in_world[:3, :3]), 'xyz')
        yaw += best_attack[1] * np.pi # depending on which patch is to be presented (0 or 1), turn attacker by 180° or not
        return T_attacker_in_world[:3, 3], roll, pitch, yaw 

    def run(self, targets, A, positions):
        offset=np.zeros(3)
        rate=2
        stretch = 9 # >1 -> slower

        all_victim_data = []
        all_attacker_data = []

        traj = Trajectory()
        # change path if needed!
        traj.loadcsv("/home/demo/ros2_ws/src/frontnet_ros/data/movey.csv")

        self.node.takeoff(targetHeight=1.0, duration=3.0)
        self.timeHelper.sleep(7.0)

        while True:
            if self.pose_a is not None and self.pose_v is not None:
                break
            self.timeHelper.sleep(0.1)

        # turn victim by -90° to face the white wall in our lab
        self.cf_v.goTo(np.array([0.0, 0.0, 1.0]), -np.pi/2., 2.)
        self.timeHelper.sleep(3.)

        start_time = self.timeHelper.time()
        while not self.timeHelper.isShutdown():
            t = self.timeHelper.time() - start_time
            if t > traj.duration * stretch:
                break

            # get new desired victim position 
            e = traj.eval(t / stretch)
            pos_v_desired = e.pos + offset

            # logging position and yaw of victim
            pos_v_current = np.array([self.pose_v.pose.position.x, self.pose_v.pose.position.y, self.pose_v.pose.position.z])
            quats_v_current = np.array([self.pose_v.pose.orientation.w, self.pose_v.pose.orientation.x, self.pose_v.pose.orientation.y, self.pose_v.pose.orientation.z])
            roll_v_current, pitch_v_current, yaw_v_current = rowan.to_euler(quats_v_current, 'xyz') 
            all_victim_data.append([t, *pos_v_current, roll_v_current, pitch_v_current, yaw_v_current,*pos_v_desired, 0., 0., 0.])

            # compute desired attacker pose
            pos_a_desired, roll_a_desired, pitch_a_desired, yaw_a_desired = self.compute_attacker_pose(pos_v_desired, targets, A, positions)


            # logging position and yaw of attacker
            pos_a_current = np.array([self.pose_a.pose.position.x, self.pose_a.pose.position.y, self.pose_a.pose.position.z])
            quats_a_current = np.array([self.pose_a.pose.orientation.w, self.pose_a.pose.orientation.x, self.pose_a.pose.orientation.y, self.pose_a.pose.orientation.z])
            roll_a_current, pitch_a_current, yaw_a_current = rowan.to_euler(quats_a_current, 'xyz') 
            all_attacker_data.append([t, *pos_a_current, roll_a_current, pitch_a_current, yaw_a_current,*pos_a_desired, roll_a_desired, pitch_a_desired, yaw_a_desired])

            # continously save data in case of crashes
            np.save(f'victim_data_{start_time}.npy', np.array(all_victim_data))
            np.save(f'attacker_data_{start_time}.npy', np.array(all_attacker_data))

            # calculate move time depending on maximum speed and angular speed
            # switching the patch takes more time than simply moving to a new setpoint!
            distance= np.linalg.norm(pos_a_current-pos_a_desired)
            angular_distance = np.abs(np.arctan2(np.sin(yaw_a_current- yaw_a_desired), np.cos(yaw_a_current - yaw_a_desired)))
            max_speed = 0.5
            max_angular_speed = 0.8
            move_time = max(distance / max_speed, 1.0, angular_distance/max_angular_speed)
            
            self.cf_a.goTo(pos_a_desired, yaw_a_desired, move_time)
            self.timeHelper.sleepForRate(rate)

        
        self.timeHelper.sleep(5.0)
        self.cf_a.notifySetpointsStop()
        self.node.land(targetHeight=0.03, duration=3.0)
        self.timeHelper.sleep(3.0)



def main():
    a = Attack()

    # load saved results for optimized patches
    # change path if needed!
    with open('/home/demo/Documents/flying_adversarial_patch/results.yaml') as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

    # get the targets the patches where optimized for
    targets = np.array([*dict['targets'].values()]).T
    
    # get which patch was assigned to which target
    A = np.array([*dict['assignment_patch']])

    # get the positions of the optimal patch for each target
    patch_in_victim = np.array([*dict['patch_in_victim'].values()]).T

    a.run(targets, A, patch_in_victim)


if __name__ == "__main__":
    main()
