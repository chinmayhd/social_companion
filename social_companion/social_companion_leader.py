"""Contains defination of all classes here."""

from typing import Tuple


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
        force[dist <= goal_threshold] = -3.0 * vel_2d[dist <= goal_threshold]
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

        for i in range(in_sqrt.shape[0]):
            if in_sqrt[0][i] < 0:
                in_sqrt[0][i] = 0

        return 0.5 * np.sqrt(in_sqrt)

    def value_r_ab(self, r_ab, speeds, desired_directions, delta_t):
        """Get the repulsive potential using AB e ^ (-b/B) formula."""
        v0 = 2.1
        sigma = 0.79
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
        lambda_importance = 0.59
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
