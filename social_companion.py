"""Contains defination of all classes here."""

import copy
import math
from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np


class Forces:
    """Contains definition of all classes here."""

    def normalize(self, vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize nx2 array along the second axis.

        input: [n,2] ndarray
        output: (normalized vectors, norm factors)
        """
        norm_factors = []
        for line in vecs:
            norm_factors.append(np.linalg.norm(line))
        norm_factors = np.array(norm_factors)
        normalized = vecs / np.expand_dims(norm_factors, -1)
        # get rid of nans
        for i in range(norm_factors.shape[0]):
            if norm_factors[i] == 0:
                normalized[i] = np.zeros(vecs.shape[1])
        return normalized, norm_factors

    def get_goal_force(self, robot_state, goal, goal_threshold):
        """Attractive force between robot and its goal."""
        relexation_time = 0.5

        vel = robot_state[0][3:5]
        yaw = robot_state[0][2]
        vel = np.array([vel[0] * np.cos(yaw), vel[0] * np.sin(yaw)])
        pos = robot_state[0][0:2]
        goal_2d = goal.reshape(1, -1)
        pos_2d = pos.reshape(1, -1)
        vel_2d = vel.reshape(1, -1)
        direction, dist = self.normalize(goal_2d - pos_2d)
        force = np.zeros((1, 2))
        # 1.2 is the max speed
        force[dist > goal_threshold] = (
            direction * np.array([1.2]) - vel_2d.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel_2d[dist <= goal_threshold]
        force /= relexation_time

        return force

    def get_obstacle_force(self, robot_state, obstacles):
        """Repulsive force between robot and obstacles."""
        sigma = 0.5
        agent_radius = 0.35
        threshold = 3 + agent_radius
        force = np.zeros((1, 2))
        if len(obstacles) == 0:
            return force
        pos = robot_state[0][0:2]
        obstacles = np.vstack(obstacles)

        diff = pos - obstacles
        directions, dist = self.normalize(diff)
        dist = dist - agent_radius
        dist_mask = dist < threshold
        directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
        force = np.sum(directions[dist_mask], axis=0)

        return force

    def desired_directions(self, robot_state, goal):
        """Get the unit vector and its magnitude."""
        destination_vectors = goal - robot_state[0][0:2]
        reshaped_vectors = destination_vectors.reshape(1, -1)
        directions, dist = self.normalize(reshaped_vectors)
        return directions, dist

    def b(self, r_ab_val, speeds, desired_directions, delta_t):
        """Calculate b.

        b denotes the semi-minor axis of the ellipse and is given by
        e: desired direction
        2b=sqrt((r_ab+(r_ab-v*delta_t*e_b))
        """
        speeds = np.array([speeds])
        speeds_b = np.expand_dims(speeds, axis=0)
        speeds_b_abc = np.expand_dims(speeds_b, axis=2)
        e_b = np.expand_dims(desired_directions, axis=0)
        in_sqrt = (
            np.linalg.norm(r_ab_val, axis=-1)
            + np.linalg.norm(r_ab_val - delta_t * speeds_b_abc * e_b, axis=-1)
        ) ** 2 - (delta_t * speeds_b) ** 2

        return 0.5 * np.sqrt(in_sqrt)

    def value_r_ab(self, r_ab, speeds, desired_directions, delta_t):
        """Get the repulsive potential using AB e ^ (-b/B) formula."""
        v0 = 2.1
        sigma = 0.3
        return v0 * np.exp(
            -self.b(
                r_ab_val=r_ab,
                speeds=speeds,
                desired_directions=desired_directions,
                delta_t=delta_t,
            )
            / sigma
        )

    def grad_r_ab(self, robot_state, ped_state, goal, delta):
        """Calculate Repulsive Potential Here."""
        robot_pos = robot_state[0][0:2]
        ped_pos = ped_state[:, :2]
        diff = ped_pos - np.expand_dims(robot_pos, 0)
        r_ab_val = np.expand_dims(diff, 0)
        robot_velocity = robot_state[0][3:5]
        init_yaw = robot_state[0][2]
        linear_velocity = robot_velocity[0]

        vx = linear_velocity * np.cos(init_yaw)
        vy = linear_velocity * np.sin(init_yaw)

        robot_velocity = np.array([vx, vy])

        robot_speed = np.linalg.norm(robot_velocity)

        desired_directions_val, _ = self.desired_directions(robot_state, goal)
        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self.value_r_ab(
            r_ab=r_ab_val,
            speeds=robot_speed,
            desired_directions=desired_directions_val,
            delta_t=delta,
        )
        dvdx = (
            self.value_r_ab(
                r_ab=r_ab_val + dx,
                speeds=robot_speed,
                desired_directions=desired_directions_val,
                delta_t=delta,
            )
            - v
        ) / delta
        dvdy = (
            self.value_r_ab(
                r_ab=r_ab_val + dy,
                speeds=robot_speed,
                desired_directions=desired_directions_val,
                delta_t=delta,
            )
            - v
        ) / delta
        return np.stack((dvdx, dvdy), axis=-1)

    def get_pedestrian_repulsive_force(self, robot_state, ped_state, goal, delta):
        """Get Repulsive Force between robot and pedestrian."""
        f_ab = -1.0 * self.grad_r_ab(robot_state, ped_state, goal=goal, delta=delta)
        F_ab = f_ab
        return np.sum(F_ab, axis=1) * 1

    def vector_angles(self, vecs: np.ndarray) -> np.ndarray:
        """Get angle using tan inverse."""
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])
        return ang

    def get_social_force(self, robot_state, ped_state):
        """Get the Social Force between pedestrian and robot."""
        lambda_importance = 0.15
        gamma = 0.5
        n = 2
        n_prime = 3

        robot_pos = robot_state[0][0:2]
        ped_pos = ped_state[:, :2]
        pos_diff = np.expand_dims(robot_pos, 0) - ped_pos
        diff_direction, diff_length = self.normalize(pos_diff)

        robot_vel = robot_state[0][3:5]
        yaw = robot_state[0][2]
        robot_vel = np.array([robot_vel[0] * np.cos(yaw), robot_vel[0] * np.sin(yaw)])

        ped_vel = ped_state[:, 2:4]
        vel_diff = -1.0 * (np.expand_dims(robot_vel, 0) - ped_vel)
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = self.normalize(interaction_vec)

        theta = self.vector_angles(interaction_direction) - self.vector_angles(
            diff_direction
        )
        B = gamma * interaction_length

        force_velocity_amount = np.exp(
            -1.0 * diff_length / B - np.square(n_prime * B * theta)
        )
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        vecs = np.fliplr(interaction_direction) * np.array([-1.0, 1.0])
        force_angle = force_angle_amount.reshape(-1, 1) * vecs

        force = force_velocity + force_angle
        force = np.sum(force.reshape((1, -1, 2)), axis=1)
        return force


class Simulator:
    """Class for simulation."""

    def __init__(
        self, robot_state, user_state, ped_state, obstacles, goal, step_width, use_plt
    ):
        """Get the values."""
        # 10 is the resolution
        resolution = 10
        self.obstacles = self.set_obstacles(obstacles, resolution=resolution)
        self.force = Forces()
        self.robot_state = robot_state
        self.user_state = user_state
        self.ped_state = ped_state
        self.final_goal = goal
        self.step_width = step_width

        self.robot_states = []
        self.user_states = []
        self.ped_states = []

        self.use_plt = use_plt
        self.max_speed = 1.1

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
            (robot_state[0], robot_state[1]), radius=0.06, color="red", fill=False
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
            (user_state[0][0], user_state[0][1]), radius=0.06, color="blue", fill=False
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
                (val[0], val[1]), radius=0.06, color="green", fill=False
            )
            plt.gca().add_artist(ped_marker)
            plt.scatter(val[0], val[1], color="green", marker="$\mathrm{P}$", s=100)

        for prev_pos in self.robot_states:
            plt.plot(prev_pos[0][0], prev_pos[0][1], "ro", alpha=0.3)
        for prev_pos in self.user_states:
            plt.plot(prev_pos[0][0], prev_pos[0][1], "bo", alpha=0.3)
        # for ped in self.ped_states:
        #     for prev_pos in ped:
        #         plt.plot(prev_pos[0], prev_pos[1], "go", alpha=0.3)
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
            [self.user_state[0][0] + 0.0, self.user_state[0][1] - 0.5]
        )
        final_goal = np.array([self.final_goal[0], self.final_goal[1] - 0.5])

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
            robot_state=self.robot_state, goal=pred_user_goal, goal_threshold=0.8
        )
        ped_repulsive_force = self.force.get_pedestrian_repulsive_force(
            robot_state=self.robot_state,
            ped_state=self.ped_state,
            goal=pred_user_goal,
            delta=1e-3,
        )
        social_force = self.force.get_social_force(
            robot_state=self.robot_state, ped_state=self.ped_state
        )
        total_force = (
            5 * obstacle_force_val
            + 0.7 * user_goal_force_val
            + 0.15 * final_goal_force_val
            + 0.5 * ped_repulsive_force
            + 3 * social_force
        )

        if self.use_plt:
            self.plt_scatter(
                robot_state=self.robot_state[0],
                user_state=self.user_state,
                pedes_state=self.ped_state,
                final_goal=self.final_goal,
                total_force=total_force,
                social_fc=social_force,
                obstacle_fc=obstacle_force_val,
                user_fc=user_goal_force_val,
                final_fc=final_goal_force_val,
                repulsive_fc=ped_repulsive_force,
            )
        return total_force

    def update_user_position(self, user_state):
        """Update position of user according to dt given."""
        self.user_state[0][0] += user_state[0][2] * self.step_width
        self.user_state[0][1] += user_state[0][3] * self.step_width
        dist_diff = np.linalg.norm(self.final_goal - self.user_state[0][:2])
        if dist_diff < 0.05:
            # print("Goal reached")
            user_state[0][2] = 0.0
            user_state[0][3] = 0.0

    def update_ped_position(self, ped_state):
        """Update position of pedestrian according to dt given."""
        self.ped_state = self.ped_state.astype(np.float64)
        self.ped_state[:, 0] += ped_state[:, 2] * self.step_width
        self.ped_state[:, 1] += ped_state[:, 3] * self.step_width

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

    def step(self, n):
        """Code starts to run here."""
        for _ in range(n):
            self.user_states.append(copy.deepcopy(self.user_state))
            self.robot_states.append(copy.deepcopy(self.robot_state))
            self.ped_states.append(copy.deepcopy(self.ped_state))
            self.update_user_position(self.user_state)
            self.update_ped_position(self.ped_state)
            total_force = self.step_once()
            self.update_robot_position(total_force)

        robot_pos_x_y = [array[0][:2] for array in self.robot_states]
        user_pos_x_y = [array[0][:2] for array in self.user_states]
        ped_pos_x_y = [arr[:, :2].tolist() for arr in self.ped_states]

        # Flatten the list
        ped_pos_x_y_flat = [item for sublist in ped_pos_x_y for item in sublist]

        robot_x_values = [values[0] for values in robot_pos_x_y]
        robot_y_values = [values[1] for values in robot_pos_x_y]
        user_x_values = [values[0] for values in user_pos_x_y]
        user_y_values = [values[1] for values in user_pos_x_y]

        # Plot
        plt.figure()
        plt.scatter(robot_x_values, robot_y_values, color="red", alpha=0.3)
        plt.scatter(user_x_values, user_y_values, color="blue", alpha=0.3)
        for val in ped_pos_x_y_flat:
            plt.scatter(val[0], val[1], color="green", alpha=0.3)
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.title("Plot of the trajectory of the robot", fontweight="bold")
        plt.scatter(-100.0, 4.0, color="red", marker="o", label="Robot")
        plt.scatter(-100.0, 3.0, color="Blue", marker="o", label="User")
        plt.scatter(-100.0, 2.0, color="green", marker="o", label="Pedestrian")
        plt.legend()
        plt.axis("equal")
        plt.xlim(-1, 5)
        plt.ylim(-2, 2)

        plt.gca().set_aspect("equal", adjustable="box")

        plt.xticks(range(-1, 6))
        plt.yticks(range(-2, 3))
        plt.grid(True)
        plt.show()
        return self

    def set_obstacles(self, obstacles, resolution):
        """Input an list of (startx, endx, starty, endy) as start and end of a line."""
        new_obstacles = []
        for startx, endx, starty, endy in obstacles:
            samples = int(np.linalg.norm((startx - endx, starty - endy)) * resolution)
            line = np.array(
                list(
                    zip(
                        np.linspace(startx, endx, samples),
                        np.linspace(starty, endy, samples),
                    )
                )
            )
            new_obstacles.append(line)
        return new_obstacles
