"""
With hindsight, we make a class to handle the actuators.
The goal is to have a nice interface to control the actuators, while 
accounting of actuator limits and noise.
"""

import numpy as np

class Actuator:
    """
    The base class for actuators and can be used across different types of 
    actuators.
    """
    def __init__(self, mounting_matrix_act_to_mc, noise_std=None, 
                 max_actuation=None, min_actuation=None):
        """
        Initialize the actuator with a maximum torque and noise standard 
        deviation.
        
        Parameters
        ----------
        mounting_matrix_act_to_mc : np.ndarray
            The mounting matrix for the actuator. The mounting matrix is a
            takes the actuator ouputs and maps them to the torques on the
            spacecraft. 
            Shape is (3, n_actuators).
        noise_std : float
            The standard deviation of the noise in the actuator.
        max_actuation : float
            The maximum torque that the actuator can produce.
        min_actuation : float
            The minimum torque that the actuator can produce.
        """

        # Check the shape matches
        assert mounting_matrix_act_to_mc.shape[0] == 3, \
            "The mounting matrix must have 3 rows, but shape is: " + \
            str(mounting_matrix_act_to_mc.shape)
        
        self.n_actuators = mounting_matrix_act_to_mc.shape[1]
        self.mounting_matrix_act_to_mc = mounting_matrix_act_to_mc
        # Use the pseudo-inverse to get the mapping from the spacecraft torques
        # to the actuator inputs.
        self.mounting_matrix_pinv_mc_to_act = \
            np.linalg.pinv(mounting_matrix_act_to_mc)

        self.noise_std = noise_std

        if max_actuation is not None:
            self.max_actuation = max_actuation
        else:
            self.max_actuation = np.inf

        if min_actuation is not None:
            self.min_actuation = min_actuation
        else:
            self.min_actuation = -np.inf

    def calculate_actuator_commands(self, torques_mc):
        """
        Given the required commanded torques (Mc), calculate the actuator 
        commands. The commands are the theoretical actuator commands.
        
        Parameters
        ----------
        torques_mc : np.ndarray
            The commanded torque on the spacecraft.

        Returns
        -------
        np.ndarray
            The actuator commands.
        """
        # Check the shape of the torques_mc
        assert torques_mc.shape == (3,), \
            "The torques_mc must have shape (3,), but shape is: " + \
            str(torques_mc.shape)

        # Calculate the actuator commands
        actuator_commands = self.mounting_matrix_pinv_mc_to_act @ torques_mc

        # Clip the actuator commands
        nearest_commands = self.clip_actuator_commands(actuator_commands)

        return nearest_commands, actuator_commands
    
    def clip_actuator_commands(self, actuator_commands):
        """
        Clip the actuator commands to the actuator limits.
        
        Parameters
        ----------
        actuator_commands : np.ndarray
            The actuator commands.

        Returns
        -------
        np.ndarray
            The clipped actuator commands.
        """

        return np.clip(actuator_commands, self.min_actuation, self.max_actuation)
    

    def sample_noise(self):
        """
        Sample noise from the noise distribution.
        
        Returns
        -------
        np.ndarray
            The sampled noise.
        """
        return np.random.normal(0, self.noise_std, self.n_actuators)


    def send_actuator_commands(self, actuator_commands, noise=True):
        """
        Send the actuator commands to the actuators. The result is the torque
        on the spacecraft.
        
        Parameters
        ----------
        actuator_commands : np.ndarray
            The actuator commands.

        Returns
        -------
        np.ndarray
            The torque on the spacecraft.
        
        np.ndarray
            The actuator commands (possibly clipped).
        """
        # Check the shape of the actuator_commands
        assert actuator_commands.shape == (self.n_actuators,), \
            "The actuator_commands must have shape (n_actuators,), but shape is: " + \
            str(actuator_commands.shape)
        
        # Add noise
        if noise and self.noise_std is not None:
            actuator_commands += self.sample_noise()

        # The actuator has limits!
        actual_actuator_commands = self.clip_actuator_commands(actuator_commands)

        # Calculate the torque on the spacecraft
        torques_mc = self.mounting_matrix_act_to_mc @ actual_actuator_commands

        return torques_mc, actual_actuator_commands
