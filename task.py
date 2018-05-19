import numpy as np
from physics_sim import PhysicsSim
import math


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 5 #Repeated call actions

        self.state_size = self.action_repeat * 6 #The value of each component in the state vector
        self.action_low = 0.01 #divide by zero
        self.action_high = 900
        self.action_size = 4
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([10., 10., 10.]) 
    

    def get_reward(self):
        
      
        reward = (1.01 -  math.tanh( abs(self.target_pos[0] - self.sim.pose[3]))) * (1.01 - math.tanh( abs(self.target_pos[1]  - self.sim.pose[4]))) * (1.01 - math.tanh( abs(self.target_pos[2]  - self.sim.pose[5])))
        
        return reward
    

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            
           
            if done :
                reward += 20
                
                
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state