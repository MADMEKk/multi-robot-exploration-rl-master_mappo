import rclpy
from rclpy.node import Node

import numpy as np
import math
from math import pi
import time

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from rclpy.qos import qos_profile_sensor_data

from start_reinforcement_learning.restart_environment import RestartEnvironment
from start_reinforcement_learning.exploration_metrics import ExplorationMetrics

class Env():
    def __init__(self, number_of_robots = 3, map_number = 1):

        self.number_of_robots = number_of_robots
        self.map_number = map_number
        self.restart_environment = RestartEnvironment(self.number_of_robots, self.map_number)
        
        # Create a list of N nodes to publish velocities to robots where N is the number of robots
        self.cmd_vel_publisher_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.cmd_vel_publisher_list[i] = PublishCMD_VEL(i)
            
        # Create a list of N nodes to read odometry from each robot (get their positions) where N is the number of robots
        self.odometry_subscriber_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.odometry_subscriber_list[i] = ReadOdom(i)
            
        # Create a list of N nodes to read laser scan information where N is the number of robots
        self.scan_subscriber_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.scan_subscriber_list[i] = ReadScan(i)
        
        # Create a node for logging output to terminal
        self.logger = Logger()
        
        # 36 rays + [angluar velocity, linear velocity, goal_x_rel, goal_y_rel]
        self.single_robot_observation_space = 40  # Updated from 38 to 40 to include goal position
        self.individual_robot_action_space = 2
        self.total_robot_observation_space = []
        for _ in range(self.number_of_robots):
            self.total_robot_observation_space.append(self.single_robot_observation_space)
        self.initGoal = True
        self.current_goal_location = []
        self.current_scan_data = 0

        self.step_counter = 0
        self.reached_goal_counter = 0
        self.total_goal_counter= 0
        self.MAX_STEPS = 500
        
        # List of robot properties
        self.current_angular_velocity = np.zeros(self.number_of_robots)
        self.current_linear_velocity = np.zeros(self.number_of_robots)
        self.current_pose_x = np.zeros(self.number_of_robots)
        self.current_pose_y = np.zeros(self.number_of_robots)
        self.current_observations = np.zeros(self.number_of_robots)

        # Robot velocity restraints 
        self.max_linear_vel = 0.6
        self.min_linear_vel = 0.05
        self.max_angular_vel = 0.5
        self.min_angular_vel = -0.5
        
        # Rewards
        self.goalReward = 20
        self.collisionReward = -20
        
        # Initialize exploration metrics
        self.exploration_metrics = ExplorationMetrics(map_size=(20.0, 20.0), resolution=0.5)
        self.exploration_metrics.reset(robot_count=self.number_of_robots)
        
        # Track goals and collisions for evaluation
        self.goals_reached = 0
        self.collisions = 0

    # get obs space.  in future we will reutrn proper box but for now just
    # return .shape E.G just a number
    def observation_space(self):
        return self.total_robot_observation_space

    def action_space(self):
        return self.individual_robot_action_space

    # Get the distance between goal and the closest robot -- DEBUGGING FUNCTION
    def getGoalDistace(self):
        closest_distance_to_goal = math.inf
        for i in range(self.number_of_robots):
            ith_robots_distance = round(math.hypot(
                self.current_goal_location[0] - self.current_pose_x[i], self.current_goal_location[1] - self.current_pose_y[i]), 2)
            if ith_robots_distance < closest_distance_to_goal:
                closest_distance_to_goal = ith_robots_distance
        return closest_distance_to_goal

    # Resize the lidar, our model has it set to 3.6 anyway but some robots have higher ones that take up too much memory
    def resize_lidar(self, scan):
        scan_range = []
        np_scan_range = np.empty(self.single_robot_observation_space, dtype= np.float32)
        # Resize scan ranges data, scan itself returns extra info about the scan, scan.ranges just gets.... the ranges
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
                np_scan_range[i] = 3.5
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
                np_scan_range[i] = 0
                print('broken')
            else:
                scan_range.append(scan.ranges[i])
                np_scan_range[i] = scan.ranges[i]
        
        return np_scan_range

    # Resets the velocities for all robots, call this after a env reset
    def reset_cmd_vel(self):
        for i in range(self.number_of_robots):
            # Reset cmd_vel and publish
            self.current_linear_velocity[i] = 0
            self.current_angular_velocity[i] = 0
            cmd_vel_publisher = self.cmd_vel_publisher_list[i]
            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = float(0)
            desired_vel_cmd.angular.z = float(0)
            cmd_vel_publisher.cmd_vel = desired_vel_cmd
            cmd_vel_publisher.pub_vel()
            
        #time.sleep(1)
    
    # Check if the given scan shows a collision
    def hasCollided(self, scan_range):
        min_range = 0.35
        # Only check the scans not the velocities indexes (They are initialised to 0 from np.empty())
        if min_range > np.min(scan_range[:35]) > 0:
            return True
        return False
    
    # Check if the current robot position shows robot has reached goal
    def hasReachedGoal(self, scan_range, robot_number):
        dis_to_goal = round(math.hypot(
                self.current_goal_location[0] - self.current_pose_x[robot_number], self.current_goal_location[1] - self.current_pose_y[robot_number]), 2)
        if dis_to_goal < 0.50:
            return True
        return False
    
    # Get basic rewards --- REWARD FUCNTION, goal and collision rewards are class attributes
    def getRewards(self):
        robotRewards = np.zeros(self.number_of_robots)
        
        # Calculate team-level exploration rewards
        newly_explored = 0
        exploration_coverage = 0
        exploration_overlap = 0
        
        # Get exploration stats if available
        if hasattr(self, 'exploration_metrics'):
            # Get robot positions for exploration updates
            robot_positions = []
            for i in range(self.number_of_robots):
                robot_positions.append((self.current_pose_x[i], self.current_pose_y[i]))
            
            # Update exploration metrics
            newly_explored = self.exploration_metrics.update_exploration(robot_positions, sensor_ranges=3.5)
            exploration_coverage = self.exploration_metrics.get_exploration_coverage()
            exploration_overlap = self.exploration_metrics.get_exploration_overlap()
            
            # Get per-robot exploration stats
            robot_stats = self.exploration_metrics.get_robot_exploration_stats()
        
        # Calculate team exploration bonus (shared among all robots)
        team_exploration_reward = newly_explored * 0.5  # Reward for newly explored cells
        
        # Calculate overlap penalty (only if there's significant overlap)
        overlap_penalty = 0
        if exploration_overlap > 0.2:  # Only penalize if overlap > 20%
            overlap_penalty = (exploration_overlap - 0.2) * 0.3
        
        for i in range(self.number_of_robots):
            # Base reward starts at a small negative value to encourage faster goal completion
            currentReward = -0.1
            
            # Calculate distance to goal
            current_distance = math.hypot(
                self.current_goal_location[0] - self.current_pose_x[i], 
                self.current_goal_location[1] - self.current_pose_y[i])
            
            # If we have a previous distance, reward getting closer to goal
            if hasattr(self, 'previous_distances') and len(self.previous_distances) > i:
                previous_distance = self.previous_distances[i]
                # Reward for moving toward goal, penalize for moving away
                distance_delta = previous_distance - current_distance
                currentReward += distance_delta * 5.0  # Scale factor to make this reward significant
            
            # Penalize being too close to obstacles (using minimum lidar reading)
            if hasattr(self, 'current_min_scan') and len(self.current_min_scan) > i:
                min_scan = self.current_min_scan[i]
                if min_scan < 0.5:  # If obstacle is close
                    obstacle_penalty = ((0.5 - min_scan) * 2.0) ** 2  # Quadratic penalty that increases as robot gets closer
                    currentReward -= obstacle_penalty
            
            # Reward for appropriate velocity - we want smooth, purposeful movement
            if self.current_linear_velocity[i] < 0.10:
                currentReward -= 0.5  # Penalize very slow movement
            elif self.current_linear_velocity[i] > 0.5:
                currentReward += 0.2  # Reward faster movement when appropriate
                
            # Penalize excessive rotation (spinning in place)
            if abs(self.current_angular_velocity[i]) > 0.3:
                currentReward -= 0.2 * abs(self.current_angular_velocity[i])
            
            # Add exploration-based rewards
            # Share team exploration reward among robots
            currentReward += team_exploration_reward / self.number_of_robots
            
            # Add individual exploration contribution if available
            if hasattr(self, 'exploration_metrics') and len(robot_stats) > i:
                # Reward based on this robot's individual exploration contribution
                individual_contribution = robot_stats[i]['exploration_percentage'] * 0.2
                currentReward += individual_contribution
            
            # Apply overlap penalty to discourage robots following each other
            currentReward -= overlap_penalty
                
            robotRewards[i] = currentReward
        
        # Store current distances for next step comparison
        self.previous_distances = [math.hypot(
            self.current_goal_location[0] - self.current_pose_x[i], 
            self.current_goal_location[1] - self.current_pose_y[i]) for i in range(self.number_of_robots)]
        
        return robotRewards                
    
    # Converts list of arrays to dictionary for MADDPG Algorithm
    def handleReturnValues(self, robotScans, robotRewards, robotDones, truncated, info):
        # Dict of each robots observation
        robot_observations = {}
        robot_rewards= {}
        robot_dones = {}
        robot_truncated = {}
        info = {}
        for i, val in enumerate(robotScans):
            robot_observations['robot'+str(i)] = val
        for i,val in enumerate(robotRewards):
            robot_rewards['robot'+str(i)] = val
        for i,val in enumerate(robotDones):
            robot_dones['robot'+str(i)] = val
        for i,val in enumerate(truncated): 
            robot_truncated['robot'+str(i)] = val
        return robot_observations, robot_rewards, robot_dones, robot_truncated, info
    
    # Updates the variables containing the robots position
    def updateRobotPosition(self):
        for i in range(self.number_of_robots):
            odom_data = None
            odom_subscriber = self.odometry_subscriber_list[i]
            while odom_data is None:
                rclpy.spin_once(odom_subscriber)
                odom_data = odom_subscriber.odom
            self.current_pose_x[i] = odom_data.pose.pose.position.x
            self.current_pose_y[i] = odom_data.pose.pose.position.y    
    
    # Adds linear and angular velocities and goal position to the scan observation
    def addVelocitiesToObs(self, scans):
        # This now adds velocities AND goal position to observations
        # Index layout: [0:35] - lidar, [36] - linear velocity, [37] - angular velocity, 
        # [38] - relative goal x, [39] - relative goal y
        velocities_index = 36  # First 36 values are lidar readings
        
        for i in range(self.number_of_robots):
            # Add velocities
            scans[i][velocities_index] = self.current_linear_velocity[i]
            scans[i][velocities_index+1] = self.current_angular_velocity[i]
            
            # Calculate relative goal position for this robot
            if self.current_goal_location and len(self.current_goal_location) >= 2:
                goal_x_rel = self.current_goal_location[0] - self.current_pose_x[i]
                goal_y_rel = self.current_goal_location[1] - self.current_pose_y[i]
                
                # Add goal position (relative to robot)
                scans[i][velocities_index+2] = goal_x_rel
                scans[i][velocities_index+3] = goal_y_rel
            else:
                # Default values if goal not set yet
                scans[i][velocities_index+2] = 0.0
                scans[i][velocities_index+3] = 0.0
                
        return scans
        
    def set_goal(self, goal_x, goal_y):
        """Set a new goal location for the robots.
        
        Args:
            goal_x: X-coordinate for the goal
            goal_y: Y-coordinate for the goal
        """
        self.current_goal_location = [goal_x, goal_y]
        # If in simulation, move the visual goal marker
        try:
            self.restart_environment.move_goal_to_position(goal_x, goal_y)
            self.logger.log(f"Goal moved to ({goal_x}, {goal_y})")
        except Exception as e:
            self.logger.log(f"Warning: Could not move goal marker: {e}")
    
    def end_of_episode_functions(self, robot_scans):
        # Quickly update position variables of robots then reset velocities
        self.updateRobotPosition()
        self.reset_cmd_vel()
        # Add the velocities to the end of observation
        self.addVelocitiesToObs(robot_scans)    
        
    # Steps the environment, (Reinforcement Learning term, it means - do this every time step)
    def step(self, action):
        self.updateRobotPosition()

        truncated = [False] * self.number_of_robots
        dones = [False] * self.number_of_robots
        info = {}

        # Read lidar scans from all robots
        robot_scans = []
        self.current_min_scan = []  # Track minimum scan distances for reward function
        
        for i in range(self.number_of_robots):
            data = None
            scan_data = self.scan_subscriber_list[i]

            while data is None:
                rclpy.spin_once(scan_data)
                data = scan_data.scan
            
            scan_range = self.resize_lidar(data)
            robot_scans.append(scan_range)
            
            # Store minimum scan distance (excluding velocity values at the end)
            min_scan = np.min(scan_range[:36]) if len(scan_range) >= 36 else 0.5
            self.current_min_scan.append(min_scan)

        # Build list of robot positions for exploration tracking
        robot_positions = []
        for i in range(self.number_of_robots):
            robot_positions.append((self.current_pose_x[i], self.current_pose_y[i]))
        
        # Update exploration metrics
        newly_explored = self.exploration_metrics.update_exploration(robot_positions, sensor_ranges=3.5)
        exploration_overlap = self.exploration_metrics.get_exploration_overlap()
        
        # Add this to info dict
        info['newly_explored'] = newly_explored
        info['exploration_coverage'] = self.exploration_metrics.get_exploration_coverage()
        info['exploration_overlap'] = exploration_overlap
        
        # Return truncated true if max steps reached
        if self.step_counter + 1 > self.MAX_STEPS:
            truncated = [True] * self.number_of_robots
            rewards = self.getRewards()
            self.end_of_episode_functions(robot_scans)
            # Return things            
            return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)
        
        # Booleans to check if episode is over
        collided = np.full(self.number_of_robots, False)
        reachedGoal = np.full(self.number_of_robots, False)

        # From the scans I want to check if The episode has terminated (any robots have crashed or reached the goal)
        for i in range(self.number_of_robots):
            collided[i] = self.hasCollided(robot_scans[i])
            reachedGoal[i] = self.hasReachedGoal(robot_scans[i], i)
        # Check if any robot has reached goal first
        if any(reachedGoal):
            self.logger.log('A robot has found the goal')
            self.reached_goal_counter+=1
            self.total_goal_counter+=1
            # If so get basic rewards quickly
            rewards = self.getRewards()
            # Get index of robots who reached goal
            indexes = np.nonzero(reachedGoal)[0]
            for idx in indexes:
                rewards[idx] += self.goalReward
                # Set done value to true for robots that reached goal
                dones[idx] = True
            
            self.end_of_episode_functions(robot_scans)
            # Return things            
            return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)
        
        # If no robots reached the goal, check if any have collided
        # Sometimes the robot accelerate too fast and their frame tilts so the sensor rays hit the floor.
        # Could adjust intertia properties of the robots, but I do think its a realistic model keeping it in
        if any(collided):
            #print('collided')
            # If so get basic rewards quickly
            rewards = self.getRewards()
            # Get index of robots who collided
            indexes = np.nonzero(collided)[0]
            for idx in indexes:
                rewards[idx] += self.collisionReward
                # Set done value to true for robots that 
                dones[idx] = True
            self.end_of_episode_functions(robot_scans)
            # Return things            
            return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)
        
        # If no robots reached goal or collided continue as normal
        
        # For each robot get its chosen action and pass it to the simulation, 
        # TODO rework action space, not happy with learning times
        for i in range(self.number_of_robots):
            name = 'robot'+str(i)
            
            # Actions
            chosen_linear_action = action[name][0]
            chosen_angular_action = action[name][1]
            
            # Increase linear velocity
            if chosen_linear_action == 2:
                self.current_linear_velocity[i] += 0.01
            # Keep linear velocity the same
            if chosen_linear_action == 1:
                self.current_linear_velocity[i] += 0.0
            # Decrease linear velocity
            if chosen_linear_action == 0:
                self.current_linear_velocity[i] -= 0.01
                
            # Increase angular velocity
            if chosen_angular_action == 2:
                self.current_angular_velocity[i] += 0.01
            # Keep angular velocity the same
            if chosen_angular_action == 1:
                self.current_angular_velocity[i] -= 0.0
            # Decrease angular velocity
            if chosen_angular_action == 0:
                self.current_angular_velocity[i] -= 0.01
            
            cmd_vel_publisher = self.cmd_vel_publisher_list[i]

            self.current_linear_velocity[i] = max(
                min(self.current_linear_velocity[i], self.max_linear_vel), self.min_linear_vel)
            self.current_angular_velocity[i] = max(
                min(self.current_angular_velocity[i], self.max_angular_vel), self.min_angular_vel)

            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = float(self.current_linear_velocity[i])
            desired_vel_cmd.angular.z = float(self.current_angular_velocity[i])

            # Publish action
            cmd_vel_publisher.cmd_vel = desired_vel_cmd
            cmd_vel_publisher.pub_vel()

        rewards = self.getRewards()
        dones = [False] * self.number_of_robots
        self.updateRobotPosition()
        # Add the velocities to the end of observation
        self.addVelocitiesToObs(robot_scans)        
        self.step_counter += 1
        return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)
    
    # Resets the environment, gets initial observations and returns robots back to there original poses
    def reset(self):
        self.step_counter = 0
        self.exploration_metrics.reset(robot_count=self.number_of_robots)
        self.restart_environment.reset_robots()
        self.updateRobotPosition()

        # When reset function is first called, we need to initialise the goal entity
        if self.initGoal:
            # spawn the goal
            self.current_goal_location = self.restart_environment.spawn_goal()
            self.initGoal = False
        
        # If robots have reached the goal node x times change location of goal
        if self.reached_goal_counter > 50:
            #time.sleep(2)
            msg = 'Found Goal, The robots have found the goal: ' + str(self.total_goal_counter) + ' times'
            self.logger.log(msg)

            self.current_goal_location = self.restart_environment.move_goal()
            self.reached_goal_counter = 0
            
        # This function wont make sense right now (could add this to observation space, or reward but it changes the whole
        # concept of only relying on lidar) ---- TODO 
        self.goal_distance = self.getGoalDistace()
        
        # Read lidar scans from all robots
        robot_scans = []
        for i in range(self.number_of_robots):
            data = None
            scan_data = self.scan_subscriber_list[i]

            while data is None:
                rclpy.spin_once(scan_data)
                #scan_data.get_logger().info("Reading data")
                data = scan_data.scan
            robot_scans.append(data)

        # Pass all lidar scans to getState function,  It resizes the lidar data and tells us if episode is done or truncated
        resized_scans = []
        print('debug1')
        for i in range(self.number_of_robots):
            rscan = robot_scans[i]
            lidar_data = self.resize_lidar(rscan)
            resized_scans.append(lidar_data)

        # Dict of each robots observation
        robot_observations = {}

        obs = self.addVelocitiesToObs(resized_scans)
        for i, val in enumerate(obs):
            robot_observations['robot'+str(i)] = val
        return robot_observations

    def get_total_unexplored_area(self):
        """
        Get the total unexplored area in the environment.
        
        Returns:
            int: Count of unexplored cells
        """
        return self.exploration_metrics.get_unexplored_area()
    
    def calculate_exploration_overlap(self):
        """
        Calculate how much exploration effort is duplicated between robots.
        
        Returns:
            float: Percentage of exploration effort that was redundant
        """
        return self.exploration_metrics.get_exploration_overlap()
    
    def get_robot_positions(self):
        """
        Get current positions of all robots.
        
        Returns:
            list: List of (x, y) positions for each robot
        """
        positions = []
        for i in range(self.number_of_robots):
            positions.append((self.current_pose_x[i], self.current_pose_y[i]))
        return positions

class ReadScan(Node):
    def __init__(self, robot_number):
        super().__init__('ReadScan'+str(robot_number))
        topic_name = "/my_bot"+str(robot_number)+"/scan"
        self.subscriber = self.create_subscription(LaserScan, topic_name, self.scan_callback,
                                                   qos_profile=qos_profile_sensor_data)
        self.scan = None

    def scan_callback(self, data):
        self.scan = data

class PublishCMD_VEL(Node):
    def __init__(self, robot_number):
        super().__init__('PublishCMD_VEL'+str(robot_number))
        topic_name = "/my_bot"+str(robot_number)+"/cmd_vel"

        self.cmd_vel_publisher = self.create_publisher(
            Twist, topic_name, 10)
        self.cmd_vel = ' '

    def pub_vel(self):
        self.cmd_vel_publisher.publish(self.cmd_vel)

class ReadOdom(Node):
    def __init__(self, robot_number):
        super().__init__('ReadOdom'+str(robot_number))
        topic_name = "/my_bot"+str(robot_number)+"/odom"
        self.subscriber = self.create_subscription(
            Odometry, topic_name, self.odom_callback, 10)
        self.odom = None

    def odom_callback(self, data):
        self.odom = data

class Logger(Node):
    def __init__(self):
        super().__init__('logger')
        
    def log(self, string):
        self.get_logger().info(string)
        


