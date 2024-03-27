"""Here we can set the robot, ped, user states."""
import copy
import logging
import math
from typing import Tuple


import numpy as np

"""Utility functions for plots and animations."""

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
except ImportError:
    plt = None
    mpl_animation = None

# Plotting code here
# scatter plot set here

total_force_plt = plt.text(-10.0, -14.0, '', fontsize=12, fontweight='bold')
social_force_plt = plt.text(-10.0, 1.0, '', fontsize=12, fontweight='bold')
obstacle_force_plt = plt.text(-10.0, 3.0, '', fontsize=12, fontweight='bold')
user_goal_plt = plt.text(-7.0, 2.0, '', fontsize=12, fontweight='bold')
final_goal_plt = plt.text(-7.0, 4.0, '', fontsize=12, fontweight='bold')
ped_repulsive_force_plt = plt.text(-12.0, 6.0, '', fontsize=12, fontweight='bold')
robot_state_plt = plt.text(2.0, -14.0, '', fontsize=12, fontweight='bold')
plt.rcParams['figure.figsize'] = [50, 50]
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot and User Movement')
plt.grid(True)
plt.ylim(-15.0, 15.0)
plt.xlim(-15.0, 20.0)
plt.scatter(-100.0, -1.0, color='red', marker='o', label='Robot')
plt.scatter(-100.0, 0.0, color='Blue', marker='o', label='User')
plt.scatter(-100.0, 0.0, color='green', marker='o', label='Pedestrian')
plt.legend()

# scatter plt code ended

logger = logging.getLogger('root')


def canvas(image_file=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_axisbelow(True)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    # fig.show()
    plt.close(fig)


def animation(length: int, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_axisbelow(True)

    context = {'ax': ax, 'update_function': None, 'init_function': None}
    yield context

    ani = mpl_animation.FuncAnimation(
        fig, init_func=context['init_function'], func=context['update_function'], frames=length, blit=True
    )
    if movie_file:
        ani.save(movie_file, writer=writer)
    # fig.show()
    plt.close(fig)


class Plotter:
    """Class for animation"""

    def __init__(
        self, ped_state, Sim_Instance, output=None, writer='imagemagick', cmap='viridis', agent_colors=None, **kwargs
    ):
        """States used for animation"""
        self.cmap = cmap
        self.agent_colors = agent_colors
        self.frames = len(Sim_Instance.robot_states)
        self.output = output
        self.writer = writer
        self.fig, self.ax = plt.subplots(**kwargs)
        self.ped_states = Sim_Instance.ped_states
        self.robot_states = Sim_Instance.robot_states
        self.obstacles = Sim_Instance.obstacles
        self.user_states = Sim_Instance.user_states
        self.ani = None

        self.robot_actors = None
        self.robot_collection = PatchCollection([])
        self.robot_collection.set(animated=True, alpha=0.6, cmap=self.cmap, clip_on=True)

        self.user_actors = None
        self.user_collection = PatchCollection([])
        self.user_collection.set(animated=True, alpha=0.6, cmap=self.cmap, clip_on=True)

        self.ped_actors = None
        self.ped_collection = PatchCollection([])
        self.ped_collection.set(animated=True, alpha=0.6, cmap=self.cmap, clip_on=True)

    def plot(self):
        self.plot_obstacles()
        self.ax.legend()
        return self.fig

    def plot_obstacles(self):
        for s in self.obstacles:
            self.ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

    def animation_init(self):
        self.plot_obstacles()
        self.ax.add_collection(self.robot_collection)
        self.ax.add_collection(self.user_collection)
        self.ax.add_collection(self.ped_collection)
        return [self.robot_collection, self.user_collection, self.ped_collection]

    def animation_update(self, i):
        self.plot_user(i)
        self.plot_robot(i)
        self.plot_ped(i)
        return [self.robot_collection, self.user_collection, self.ped_collection]

    def animate(self):
        """Animation method."""
        self.ani = mpl_animation.FuncAnimation(
            self.fig, init_func=self.animation_init, func=self.animation_update, frames=self.frames, blit=True
        )
        return self.ani
    
    def __enter__(self):
        logger.info('Start Plotting')
        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle='dotted')
        self.ax.set_aspect('equal')
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')

        plt.rcParams['animation.html'] = 'jshtml'
        # set the gif size here
        self.ax.set(xlim=(-10, 20), ylim=(-5, 5))
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            logger.error(
                f'Exception type: {exception_type}; Exception value: {exception_value}; Traceback: {traceback}'
            )
        logger.info('Plotting ends.')
        if self.output:
            if self.ani:
                output = self.output + '.gif'
                logger.info(f'Saving animation as {output}')
                self.ani.save(output, writer=self.writer)
            else:
                output = self.output + '.png'
                logger.info(f'Saving plot as {output}')
                self.fig.savefig(output, dpi=300)
        plt.close(self.fig)

    def plot_robot(self, step=-1):
        """Generate patches for robot.

        :param step: index of state, default is the latest
        :return: list of patches
        """

        # print("ROBOT STATES",self.robot_states)
        current_state = self.robot_states[step]
        # print("CURRENT STATE",current_state)
        current_state = np.array(current_state)
        # radius = 0.2 + np.linalg.norm(current_state[:, 2:4], axis=-1) / 2.0 * 0.3
        radius = np.array([0.2])
        if self.robot_actors:
            for i, robot in enumerate(self.robot_actors):
                robot.center = current_state[i, :2]
                robot.set_radius(0.2)
        else:
            self.robot_actors = [Circle(pos, radius=r) for pos, r in zip(current_state[:, :2], radius)]
        self.robot_collection.set_paths(self.robot_actors)
        self.robot_collection.set_facecolor('red')

    def plot_user(self, step=-1):
        """Generate patches for user

        :param step: index of state, default is the latest
        :return: list of patches
        """
        # print("USER STATES", self.user_states)
        current_state = self.user_states[step]
        # print("Current user _state here",current_state)
        # radius = 0.2 + np.linalg.norm(current_state[:, 2:4], axis=-1) / 2.0 * 0.3
        radius = np.array([0.2])
        if self.user_actors:
            for i, user in enumerate(self.user_actors):
                user.center = current_state[i, :2]
                user.set_radius(0.2)
        else:
            self.user_actors = [Circle(pos, radius=r) for pos, r in zip(current_state[:, :2], radius)]
        self.user_collection.set_paths(self.user_actors)
        self.user_collection.set_facecolor('purple')

    def plot_ped(self, step=-1):
        """Generate patches for pedestrians

        :param step: index of state, default is the latest
        :return: list of patches
        """
        current_state = self.ped_states[step]
        radius = [0.2] * len(current_state)
        if self.ped_actors:
            for i, ped in enumerate(self.ped_actors):
                ped.center = current_state[i, :2]
                ped.set_radius(0.2)
        else:
            self.ped_actors = [Circle(pos, radius=r) for pos, r in zip(current_state[:, :2], radius)]
        self.ped_collection.set_paths(self.ped_actors)
        self.ped_collection.set_facecolor('green')


# Plotting code ended here

# Class which contains all forces
class Forces:
    def normalize(self, vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize nx2 array along the second axis

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

    def get_final_goal_force(self, robot_state, goal):
        # May have to change these factors
        relexation_time = 0.5
        goal_threshold = 0.1

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
        force[dist > goal_threshold] = (direction * np.array([1.2]) - vel_2d.reshape((-1, 2)))[
            dist > goal_threshold, :
        ]
        force[dist <= goal_threshold] = -1.0 * vel_2d[dist <= goal_threshold]
        force /= relexation_time

        return force

    def get_obstacle_force(self, robot_state, obstacles):
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

    def r_ab(self, robot_state, ped_state):
        robot_state = robot_state[0][0:2]
        ped_state = ped_state[:, :2]
        diff = np.expand_dims(robot_state, 0) - ped_state

        diff = np.expand_dims(diff, 0)
        return diff

    def desired_directions(self, robot_state, goal):
        destination_vectors = goal - robot_state[0][0:2]
        reshaped_vectors = destination_vectors.reshape(1, -1)
        directions, dist = self.normalize(reshaped_vectors)

        # print("directions",directions)
        return directions, dist

    def b(self, r_ab_val, speeds, desired_directions, delta_t):
        """Calculate b.

        b denotes the semi-minor axis of the ellipse and is given by
        e: desired direction
        2b=sqrt((r_ab+(r_ab-v*delta_t*e_b))
        """
        # print(delta_t)
        speeds = np.array([speeds])
        speeds_b = np.expand_dims(speeds, axis=0)
        speeds_b_abc = np.expand_dims(speeds_b, axis=2)
        e_b = np.expand_dims(desired_directions, axis=0)
        in_sqrt = (
            np.linalg.norm(r_ab_val, axis=-1) + np.linalg.norm(r_ab_val - delta_t * speeds_b_abc * e_b, axis=-1)
        ) ** 2 - (delta_t * speeds_b) ** 2
        return 0.5 * np.sqrt(in_sqrt)

    def value_r_ab(self, r_ab, speeds, desired_directions, delta_t):
        v0 = 2.1
        sigma = 0.3
        return v0 * np.exp(
            -self.b(r_ab_val=r_ab, speeds=speeds, desired_directions=desired_directions, delta_t=delta_t) / sigma
        )

    def grad_r_ab(self, robot_state, ped_state, goal, delta):
        r_ab_val = self.r_ab(robot_state, ped_state)
        robot_velocity = robot_state[0][3:5]
        init_yaw = robot_state[0][2]
        linear_velocity = robot_velocity[0]

        vx = linear_velocity * np.cos(init_yaw)
        vy = linear_velocity * np.sin(init_yaw)

        robot_velocity = np.array([vx, vy])

        robot_speed = np.linalg.norm(robot_velocity)
        speeds = np.linalg.norm(robot_speed)

        desired_directions_val, _ = self.desired_directions(robot_state, goal)
        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self.value_r_ab(r_ab=r_ab_val, speeds=speeds, desired_directions=desired_directions_val, delta_t=delta)
        dvdx = (
            self.value_r_ab(
                r_ab=r_ab_val + dx, speeds=speeds, desired_directions=desired_directions_val, delta_t=delta
            )
            - v
        ) / delta
        dvdy = (
            self.value_r_ab(
                r_ab=r_ab_val + dy, speeds=speeds, desired_directions=desired_directions_val, delta_t=delta
            )
            - v
        ) / delta
        return np.stack((dvdx, dvdy), axis=-1)

    def get_pedestrian_repulsive_force(self, robot_state, ped_state, goal, delta):
        f_ab = -1.0 * self.grad_r_ab(robot_state, ped_state, goal=goal, delta=delta)
        F_ab = f_ab
        return np.sum(F_ab, axis=1) * 1

    def vector_angles(self, vecs: np.ndarray) -> np.ndarray:
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
        return ang

    def left_normal(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
        return vecs

    def get_social_force(self, robot_state, ped_state):
        lambda_importance = 2.0
        gamma = 0.35
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

        theta = self.vector_angles(interaction_direction) - self.vector_angles(diff_direction)
        B = gamma * interaction_length

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(-1.0 * diff_length / B - np.square(n * B * theta))
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * self.left_normal(interaction_direction)

        force = force_velocity + force_angle  # n*(n-1) x 2
        force = np.sum(force.reshape((1, -1, 2)), axis=1)
        return force


"""Field of view computation."""


class FieldOfView(object):
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.
    """

    def __init__(self, phi=None, out_of_view_factor=None):
        phi = phi or 1.0
        out_of_view_factor = out_of_view_factor or 0.5
        self.cosphi = np.cos(phi / 180.0 * np.pi)
        self.out_of_view_factor = out_of_view_factor

    def __call__(self, desired_direction, forces_direction):
        """Weighting factor for field of view.

        desired_direction : e, rank 2 and normalized in the last index.
        forces_direction : f, rank 3 tensor.
        """
        in_sight = (
            np.einsum('aj,abj->ab', desired_direction, forces_direction)
            > np.linalg.norm(forces_direction, axis=-1) * self.cosphi
        )
        out = self.out_of_view_factor * np.ones_like(in_sight)
        out[in_sight] = 1.0
        np.fill_diagonal(out, 0.0)
        return out


class Simulator:
    def __init__(self, robot_state, user_state, ped_state, obstacles, goal, step_width, use_plt):
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

        plt.scatter(robot_state[0], robot_state[1], color='red', marker='o')
        plt.scatter(user_state[0][0], user_state[0][1], color='blue', marker='o')
        for val in pedes_state:
            plt.scatter(val[0], val[1], color='green', marker='o')
        plt.text(final_goal[0], final_goal[1], 'G', fontsize=12, ha='right')
        total_force_plt.set_text(f'Total Force: {total_force}')
        social_force_plt.set_text(f'Social_force: {social_fc}')
        obstacle_force_plt.set_text(f'obstacle_force: {obstacle_fc}')
        user_goal_plt.set_text(f'user_goal: {user_fc}')
        final_goal_plt.set_text(f'final_goal: {final_fc}')
        robot_state_plt.set_text(f'Robot state: {robot_state}')
        ped_repulsive_force_plt.set_text(f'Repulsive Force : {repulsive_fc}')
        plt.pause(0.1)  # Pause to update the plot

    def step_once(self):
        pred_user_goal = np.array([self.user_state[0][0] + 0.0, self.user_state[0][1] - 0.5])
        if np.linalg.norm(self.user_state[0, 0:2] - self.robot_state[0, 0:2]) < 0.05:
            print(122)
        obstacle_force_val = self.force.get_obstacle_force(robot_state=self.robot_state, obstacles=self.obstacles)
        final_goal_force_val = self.force.get_final_goal_force(robot_state=self.robot_state, goal=self.final_goal)
        user_goal_force_val = self.force.get_final_goal_force(robot_state=self.robot_state, goal=pred_user_goal)
        ped_repulsive_force = self.force.get_pedestrian_repulsive_force(
            robot_state=self.robot_state, ped_state=self.ped_state, goal=pred_user_goal, delta=1e-3
        )
        social_force = self.force.get_social_force(robot_state=self.robot_state, ped_state=self.ped_state)
        # These are the alpha, beta, gamma values from the paper. Dont know which one is perfect
        total_force = (
            0.75 * obstacle_force_val
            + 0.4 * user_goal_force_val
            + 0.1 * final_goal_force_val
            + 1 * ped_repulsive_force
            + 4 * social_force
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
        self.user_state[0][0] += user_state[0][2] * self.step_width
        self.user_state[0][1] += user_state[0][3] * self.step_width

        # distance from goal
        dist_diff = np.linalg.norm(self.final_goal - self.user_state[0][:2])
        if dist_diff < 0.5:
            # print("Goal reached")
            user_state[0][2] = 0.0
            user_state[0][3] = 0.0

    def update_ped_position(self, ped_state):
        self.ped_state = self.ped_state.astype(np.float64)
        self.ped_state[:, 0] += ped_state[:, 2] * self.step_width
        self.ped_state[:, 1] += ped_state[:, 3] * self.step_width

    def capped_velocity(self, desired_velocity, max_velocity):
        desired_velocity = desired_velocity.reshape(1, -1)
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    def update_robot_position(self, total_force):
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

        # print("VEL",vel)

        yaw = math.atan2(vel[1], vel[0])
        self.robot_state[0][0] += vel[0] * self.step_width
        self.robot_state[0][1] += vel[1] * self.step_width

        self.robot_state[0][2] = yaw

        linear_velocity = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
        # angular_velocity = (yaw - init_yaw) * math.pi / self.step_width
        # Calculate the change in yaw
        delta_yaw = yaw - init_yaw

        # Ensure delta_yaw is within the range of -pi to pi
        delta_yaw = ((delta_yaw + np.pi) % (2 * np.pi)) - np.pi

        # Calculate the elapsed time (assuming self.step_width is the time step)
        delta_time = self.step_width

        # Calculate angular velocity
        angular_velocity = delta_yaw / delta_time

        self.robot_state[0][3] = linear_velocity
        self.robot_state[0][4] = angular_velocity

    def step(self, n):
        for _ in range(n):
            self.user_states.append(copy.deepcopy(self.user_state))
            self.robot_states.append(copy.deepcopy(self.robot_state))
            self.ped_states.append(copy.deepcopy(self.ped_state))
            self.update_user_position(self.user_state)
            self.update_ped_position(self.ped_state)
            total_force = self.step_once()
            self.update_robot_position(total_force)
        return self

    def set_obstacles(self, obstacles, resolution):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        new_obstacles = []
        for startx, endx, starty, endy in obstacles:
            samples = int(np.linalg.norm((startx - endx, starty - endy)) * resolution)
            line = np.array(list(zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))))
            new_obstacles.append(line)
        return new_obstacles
