"""Here we can set the robot, ped, user states.

The forces calculation is done in the social_companion.py.

Class Forces contains the defination for all of the forces used
"""

import numpy as np

import social_companion as sc

if __name__ == "__main__":
    robot_state = np.array([[0.0, -0.5, np.deg2rad(0.0), 0.0, np.deg2rad(0.0)]])
    user_state = np.array([[0.0, 0.0, 1, 0.0]])
    # Passing
    ped_state = np.array(
        [
            [3, -0.5, 0, 0],
        ]
    )
    goal = np.array([4, -0])
    obs = []

    sim = sc.Simulator(
        robot_state=robot_state,
        user_state=user_state,
        ped_state=ped_state,
        step_width=0.05,
        obstacles=obs,
        goal=goal,
        use_plt=True,
    )
    sim.step(200)
