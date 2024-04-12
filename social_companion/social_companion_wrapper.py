"""Ros2 Wrapper for social companion."""

import copy
import math

from common_python.nodes.base_class import BaseLifecycleNode
from common_python.nodes.config import Topic

from geometry_msgs.msg import Point

import matplotlib.pyplot as plt

import numpy as np

from geometry_msgs.msg import (
    Point,
    Twist,
)

import rclpy

from social_companion.social_companion_leader import Forces

from visualization_msgs.msg import Marker, MarkerArray


class social_companion_ros_wrapper(BaseLifecycleNode):
    def __init__(self):
        super().__init__("social_companion_ros_wrapper", state_subscribe=True)

        try:
            self.step_width = 0.05
            self.use_plt = False
            self.debug = True
            self.timer_period = 0.05
            # Dummy values initially
            self.user_state = np.array([[-0.0, 0.0, 0.5, 0.0, 8.0, 0]])
            self.robot_state = np.array(
                [[-0.0, -0.5, np.deg2rad(0.0), 0.0, np.deg2rad(0.0)]]
            )

            self.ped_state = np.array(
                [
                    [3.0, 2.7, 0, -0.6, 2.0, -1.5],
                ]
            )

            self.max_speed = 1.1
            self.obs = [[1, 2, -2, 2]]

            self.robot_states = []
            self.user_states = []
            self.ped_states = []

            self.obstacles = []

            self.force = Forces()
            self.robot_pub = self.create_publisher(Marker, "/debug/robot", 1)
            self.user_pub = self.create_publisher(Marker, "/debug/user", 1)
            self.ped_pub = self.create_publisher(MarkerArray, "/debug/ped", 1)
            self.robot_line_pub = self.create_publisher(Marker, "/debug/linestrip", 1)
            self.goal_publisher = self.create_publisher(Marker, "/debug/goal", 10)
            self.emergency_flag = 0
            self.cmd_publisher = self.create_publisher(
                Topic.CONTROL_COMMAND.typ,
                Topic.CONTROL_COMMAND.name,
                qos_profile=Topic.CONTROL_COMMAND.qos,
            )
            self.debug_cmd_publisher = self.create_publisher(
                Topic.CONTROL_COMMAND.typ,
                "/debug/control_command",
                qos_profile=Topic.CONTROL_COMMAND.qos,
            )
            self.prev_filtered_cmd = np.zeros(2)
            self.prev_cmd = np.zeros(2)

            self.robot_line_marker_msg = Marker()
            self.robot_line_marker_msg.points = []
            self.use_user_callback = True

            self.timer = self.create_timer(self.timer_period, self.timer_callback)
            # self.create_subscription(
            #     ObjectResult,
            #     "/tracked_object",
            #     self.pedestrians_callback,
            #     10,
            # )

            self.create_subscription(
                Topic.TRACKED_OBJECT_FILTERED.typ,
                Topic.TRACKED_OBJECT_FILTERED.name,
                self.pedestrians_callback,
                Topic.TRACKED_OBJECT_FILTERED.qos,
            )

            self.create_subscription(
                Topic.EMERGENCY_FLAG.typ,
                Topic.EMERGENCY_FLAG.name,
                self.emergency_callback,
                qos_profile=Topic.EMERGENCY_FLAG.qos,
            )
        except Exception as err:
            self.error_transition(err)

    def emergency_callback(self, msg):
        """Emergency brake."""
        # 0:normal 1:emergency(long_vel) 2:emergency(lat_vel & long_vel)
        self.emergency_flag = msg.data

    def pedestrians_callback(self, msg):
        """Get user and pedestrian state."""
        assert msg.header.frame_id == "odom"
        self.set_pedestrian_msg(stamp=msg.header.stamp, infos=msg.infos)

    def set_pedestrian_msg(self, stamp, infos):
        """Set state."""
        self.pedes_state = []
        for info in infos:
            if info.is_user:
                self.set_user_msg(info=info)
            else:
                self.pedes_state.append(self.set_info_as_states(info))

        self.pedes_state = np.array(self.pedes_state)
        self.ped_state = self.pedes_state
        self.is_ok_pedestrians = True

    def set_user_msg(self, info):
        self.user_state = self.set_info_as_states(info)
        self.user_state = np.array(self.user_state).reshape(1, -1)
        if not self.use_user_callback:
            self.is_ok_user = True

    def set_info_as_states(self, info):
        position_o = np.array([info.position[0].x, info.position[0].y])
        vel_o = np.array([info.velocity[0].x, info.velocity[0].y])

        vel_angle = np.arctan2(vel_o[1], vel_o[0])
        vel_norm = np.linalg.norm(vel_o)
        filtered_angle = vel_angle

        # velocity filtering
        filtered_speed = vel_norm
        vel_o = np.array(
            [
                filtered_speed * np.cos(filtered_angle),
                filtered_speed * np.sin(filtered_angle),
            ]
        )
        goal_o = position_o + vel_o * 10.0

        result = np.array(
            np.hstack((position_o, vel_o, goal_o)).tolist()
        )  # [x, y, vx, vy, gx, gy]
        return result

    def publish_marker(self, robot_pos, user_pos, ped_pos):
        robot_marker = Marker()
        robot_marker.header.frame_id = "odom"
        robot_marker.type = Marker.SPHERE_LIST
        robot_marker.action = Marker.ADD
        robot_marker.scale.x = 0.5
        robot_marker.scale.y = 0.5
        robot_marker.scale.z = 1.0
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 0.0
        robot_marker.color.a = 1.0
        point = Point()
        point.x = robot_pos[0]
        point.y = robot_pos[1]
        point.z = 0.0
        robot_marker.points.append(point)
        self.robot_pub.publish(robot_marker)

        user_marker = Marker()
        user_marker.header.frame_id = "odom"
        user_marker.type = Marker.SPHERE_LIST
        user_marker.action = Marker.ADD
        user_marker.scale.x = 0.2
        user_marker.scale.y = 0.2
        user_marker.scale.z = 1.0
        user_marker.color.r = 0.0
        user_marker.color.g = 0.0
        user_marker.color.b = 1.0
        user_marker.color.a = 1.0
        point = Point()
        point.x = user_pos[0]
        point.y = user_pos[1]
        point.z = 0.0
        user_marker.points.append(point)
        self.user_pub.publish(user_marker)

        self.robot_line_marker_msg.header.frame_id = "odom"
        self.robot_line_marker_msg.ns = "line_strip"
        self.robot_line_marker_msg.id = 0
        self.robot_line_marker_msg.type = Marker.LINE_STRIP
        self.robot_line_marker_msg.action = Marker.ADD
        self.robot_line_marker_msg.pose.orientation.w = 1.0
        self.robot_line_marker_msg.scale.x = 0.05
        self.robot_line_marker_msg.color.r = 1.0
        self.robot_line_marker_msg.color.a = 1.0

        point = Point()
        point.x = robot_pos[0]
        point.y = robot_pos[1]
        point.z = 0.0
        self.robot_line_marker_msg.points.append(point)

        self.robot_line_pub.publish(self.robot_line_marker_msg)

        ped_array = MarkerArray()
        for idx, position in enumerate(ped_pos):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            pos = Point()
            pos.x = position[0]
            pos.y = position[1]
            marker.pose.position = pos
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            ped_array.markers.append(marker)

        self.ped_pub.publish(ped_array)

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = Point(x=6.0, y=0.0, z=0.0)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.5
        marker.text = "G"
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        self.goal_publisher.publish(marker)

    def update_user_position(self, user_state):
        """Update position of user according to step width given."""
        self.user_state[0][0] += user_state[0][2] * self.step_width
        self.user_state[0][1] += user_state[0][3] * self.step_width
        final_goal = self.user_state[0][4:6]
        dist_diff = np.linalg.norm(final_goal - self.user_state[0][:2])
        if dist_diff < 0.05:
            # print("Goal reached")
            user_state[0][2] = 0.0
            user_state[0][3] = 0.0

    def update_ped_position(self, ped_state):
        """Update position of pedestrian according to step width given."""
        self.ped_state = self.ped_state.astype(np.float64)
        self.ped_state[:, 0] += ped_state[:, 2] * self.step_width
        self.ped_state[:, 1] += ped_state[:, 3] * self.step_width
        for i in range(len(self.ped_state)):
            final_goal = self.ped_state[i, 4:6]
            current_position = self.ped_state[i, :2]
            dist_diff = np.linalg.norm(final_goal - current_position)
            if dist_diff < 0.05:
                self.ped_state[i, 2] = 0.0
                self.ped_state[i, 3] = 0.0

    def capped_velocity(self, desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_velocity = desired_velocity.reshape(1, -1)
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    def update_robot_position(self, total_force):
        """Update position of robot according to dt given and the force calculated."""
        init_yaw = self.robot_state[0][2]
        vel = self.robot_state[0][3:5]

        linear_velocity = vel[0]
        angular_velocity = vel[1]

        vx = linear_velocity * np.cos(init_yaw)
        vy = linear_velocity * np.sin(init_yaw)
        vel = np.array([vx, vy])
        vel += total_force[0] * self.step_width
        vel = self.capped_velocity(vel, self.max_speed)
        vel = vel[0]
        yaw = math.atan2(vel[1], vel[0])
        self.robot_state[0][0] += vel[0] * self.step_width
        self.robot_state[0][1] += vel[1] * self.step_width

        self.robot_state[0][2] = yaw

        linear_velocity = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
        delta_yaw = yaw - init_yaw
        delta_yaw = ((delta_yaw + np.pi) % (2 * np.pi)) - np.pi
        delta_time = self.step_width
        angular_velocity = delta_yaw / delta_time

        self.robot_state[0][3] = linear_velocity
        self.robot_state[0][4] = angular_velocity

    def plt_scatter(
        self,
        robot_state,
        user_state,
        pedes_state,
        total_force,
        social_fc,
        obstacle_fc,
        user_fc,
        final_fc,
        repulsive_fc,
        final_goal,
    ):
        """Plot everything."""
        plt.cla()

        robot_marker = plt.Circle(
            (robot_state[0], robot_state[1]), radius=0.4, color="red", fill=False
        )
        plt.gca().add_artist(robot_marker)
        plt.plot(
            robot_state[0],
            robot_state[1],
            color="red",
            marker="$\mathrm{R}$",
            markersize=8,
        )
        user_marker = plt.Circle(
            (user_state[0][0], user_state[0][1]), radius=0.2, color="blue", fill=False
        )
        plt.gca().add_artist(user_marker)
        plt.plot(
            user_state[0][0],
            user_state[0][1],
            color="blue",
            marker="$\mathrm{U}$",
            markersize=8,
        )
        plt.plot(final_goal[0], final_goal[1], "xk", markersize=8)
        self.plot_arrow(1.0 * robot_state[0], 1.0 * robot_state[1], robot_state[2])
        for val in pedes_state:
            ped_marker = plt.Circle(
                (val[0], val[1]), radius=0.2, color="green", fill=False
            )
            plt.gca().add_artist(ped_marker)
            plt.scatter(val[0], val[1], color="green", marker="$\mathrm{P}$", s=100)

        for prev_pos in self.robot_states:
            plt.plot(prev_pos[0][0], prev_pos[0][1], "ro", alpha=0.3)
        for prev_pos in self.user_states:
            plt.plot(prev_pos[0][0], prev_pos[0][1], "bo", alpha=0.3)
        for ped in self.ped_states:
            for prev_pos in ped:
                plt.plot(prev_pos[0], prev_pos[1], "go", alpha=0.3)
        plt.title("Social Companion Passing Pedestrian Scenario", fontweight="bold")
        plt.axis("equal")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.xlim(-1, 5)
        plt.ylim(-2, 3)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xticks(range(-1, 6))
        plt.yticks(range(-2, 4))
        plt.scatter(-100.0, 4.0, color="red", marker="$\mathrm{R}$", label="Robot")
        plt.scatter(-100.0, 3.0, color="blue", marker="$\mathrm{U}$", label="User")
        plt.scatter(
            -100.0, 2.0, color="green", marker="$\mathrm{P}$", label="Pedestrian"
        )
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)

    def step_once(self):
        """Calculate all Forces."""
        pred_user_goal = np.array(
            [self.user_state[0][0] + 0.0, self.user_state[0][1] - 1.0]
        )

        final_goal = self.user_state[0][4:6]
        final_goal = np.array([final_goal[0], final_goal[1] - 1.0])

        if np.linalg.norm(final_goal - self.robot_state[0, 0:2]) < 0.05:
            self.robot_state[0][3] = 0
            self.robot_state[0][4] = 0
        obstacle_force_val = self.force.get_obstacle_force(
            robot_state=self.robot_state, obstacles=self.obstacles
        )
        final_goal_force_val = self.force.get_goal_force(
            robot_state=self.robot_state, goal=final_goal, goal_threshold=0.8
        )
        user_goal_force_val = self.force.get_goal_force(
            robot_state=self.robot_state, goal=pred_user_goal, goal_threshold=0.2
        )
        ped_repulsive_force = self.force.get_pedestrian_repulsive_force(
            robot_state=self.robot_state,
            ped_state=self.ped_state,
            goal=pred_user_goal,
            delta=1,
        )
        social_force = self.force.get_social_force(
            robot_state=self.robot_state, ped_state=self.ped_state
        )
        user_social_force = self.force.get_social_force(
            robot_state=self.robot_state, ped_state=self.user_state
        )
        user_repulsive_force = self.force.get_pedestrian_repulsive_force(
            robot_state=self.robot_state,
            ped_state=self.user_state,
            goal=pred_user_goal,
            delta=1,
        )
        total_force = (
            5 * obstacle_force_val
            + 1 * user_goal_force_val
            + 0.15 * final_goal_force_val
            + 0.2 * ped_repulsive_force
            + 9 * social_force
            + 1 * user_social_force
        )

        if self.use_plt:
            self.plt_scatter(
                robot_state=self.robot_state[0],
                user_state=self.user_state,
                pedes_state=self.ped_state,
                final_goal=self.user_state[0][4:6],
                total_force=total_force,
                social_fc=social_force,
                obstacle_fc=obstacle_force_val,
                user_fc=user_goal_force_val,
                final_fc=final_goal_force_val,
                repulsive_fc=ped_repulsive_force,
            )

        if self.debug:
            self.publish_marker(
                self.robot_state[0][0:2],
                self.user_state[0][0:2],
                self.ped_state[:, 0:2],
            )
        return total_force

    def plot_arrow(self, x, y, yaw, length=0.1, head_length=0.025, head_width=0.025):
        """Draw arrow to visualize yaw."""
        plt.arrow(
            x,
            y,
            length * math.cos(yaw),
            length * math.sin(yaw),
            head_length=head_length,
            head_width=head_width,
        )
        plt.plot(x, y)

    def lowpass_filter(self, prev_output, input, prev_input, t=0.5):
        ts = self.timer_period
        k = 1.0
        y = (2.0 * t - ts) / (2.0 * t + ts) * prev_output + k * ts / (2.0 * t + ts) * (
            input + prev_input
        )

        return y

    def modify_action_with_emergency_flag(self, vx, wz, emergency_flag):
        if emergency_flag == 0:
            return [vx, wz]
        elif emergency_flag in [1, 2, 3]:
            self.prev_filtered_cmd = np.zeros(2)
            self.prev_cmd = np.zeros(2)
            return [0.0, 0.0]
        else:
            raise NotImplementedError()

    def filtering_action(self, action):
        # if self.turn_flag:
        #     action = self.turn_cmd[:]
        #     self.turn_cmd = [0.0, 0.0]

        # filtering vx
        self.prev_filtered_cmd[0] = self.lowpass_filter(
            prev_output=self.prev_filtered_cmd[0],
            input=action[0],
            prev_input=self.prev_cmd[0],
        )
        # filtering wz
        self.prev_filtered_cmd[1] = self.lowpass_filter(
            prev_output=self.prev_filtered_cmd[1],
            input=action[1],
            prev_input=self.prev_cmd[1],
        )

        debug_msg = Twist()
        debug_msg.linear.x = self.prev_filtered_cmd[0]
        debug_msg.angular.z = self.prev_filtered_cmd[1]
        self.debug_cmd_publisher.publish(debug_msg)

        self.prev_cmd = action[:]
        action = self.prev_filtered_cmd[:]

        # modification for stopping
        action = self.modify_action_with_emergency_flag(
            action[0], action[1], self.emergency_flag
        )

        if action[0] < 0:
            action[0] = 0.0
        return action

    def timer_callback(self):
        try:
            self.user_states.append(copy.deepcopy(self.user_state))
            self.robot_states.append(copy.deepcopy(self.robot_state))
            self.ped_states.append(copy.deepcopy(self.ped_state))
            total_force = self.step_once()
            self.update_robot_position(total_force)
            action = self.robot_state[0][3:5]
            action = self.filtering_action(action)
            msg = Twist()
            msg.linear.x = action[0]
            msg.angular.z = action[1]
            self.cmd_publisher.publish(msg)

        except Exception as err:
            self.error_transition(err)


def main(args=None):
    rclpy.init(args=args)
    sc = social_companion_ros_wrapper()

    rclpy.spin(sc)

    sc.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
