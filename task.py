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
        self.target_pos = target_pos if target_pos is not None else np.array([15., 15., 15.]) 
    

    def get_reward(self):
        
      
        reward = (1.1 -  math.tanh( abs(self.target_pos[0] - self.sim.pose[0]))) * (1.1 - math.tanh( abs(self.target_pos[1]  - self.sim.pose[1]))) * (1.1 - math.tanh( abs(self.target_pos[2]  - self.sim.pose[2])))
        reward = -0.1 * abs(self.sim.pose[3:5]).sum()    #penalise non zero pitch and roll to keep broadly level
        return reward
    
    def get_reward1(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # refine reward to encourage flying for as long as possible and favour accuracy in z slightly
        #reward = 25 - 0.1*abs(self.sim.pose[:2] - self.target_pos[:2]).sum() \
        #    - 5*abs(self.sim.pose[2] - self.target_pos[2]) \
        #    - 0.1*abs(self.sim.v[3:6]).sum() \
        #    - 1*abs(self.sim.angular_v[0:2]).sum()
        #reward = 1 - abs(self.sim.pose[2] - self.target_pos[2]) - 0.02 * abs(self.sim.angular_v[0:2]).sum()
        
        reward = -0.5 + 1.1 * np.exp(0-(self.sim.pose[2] - self.target_pos[2])**2 / 64) \
            +0.15 * np.exp(0-(self.sim.pose[0] - self.target_pos[0])**2 / 81)  \
            +0.15 * np.exp(0-(self.sim.pose[1] - self.target_pos[1])**2 / 81)  \
            -0.01 * ((self.sim.pose[3])**2 + (self.sim.pose[4])**2)
            #- 0.1 * abs(self.sim.pose[3:5]).sum()    #penalise non zero pitch and roll to keep broadly level
            #- 0.05 * abs(self.sim.v[3:6]).sum()       \
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