"""Here we can set the robot, ped, user states.

The forces calculation is done in the social_companion.py.

Class Forces contains the defination for all of the forces used
"""

import numpy as np

import social_companion as sc

if __name__ == '__main__':
    robot_state = np.array([[-4.0, -0.5, np.deg2rad(0.0), 0.0, np.deg2rad(0.0)]])
    # User state, ped_state = [px py vx vy]
    user_state = np.array([[-4.0, 0.0, 1, 0.0]])
    ped_state = np.array(
        [
            [3, -0.5, 0, 0],
            [10, -0.5, 0, 0],
            [10, -2, 0, 0],
        ]
    )
    goal = np.array([15.0, -0])
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[1,2, -1, 1]]
    obs = []

    sim = sc.Simulator(
        robot_state=robot_state,
        user_state=user_state,
        ped_state=ped_state,
        step_width=0.2,
        obstacles=obs,
        goal=goal,
        use_plt=True,
    )
    # print(sim.user_states)
    sim.step(200)
