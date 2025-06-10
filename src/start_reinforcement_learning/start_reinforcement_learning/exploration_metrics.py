import numpy as np
import math

class ExplorationMetrics:
    """
    A class for tracking exploration coverage and efficiency metrics.
    This class maintains a grid representation of the environment and tracks
    which cells have been explored by each robot.
    """
    
    def __init__(self, map_size=(20.0, 20.0), resolution=0.5):
        """
        Initialize the exploration metrics tracker.
        
        Args:
            map_size: Tuple (width, height) representing the map dimensions in meters
            resolution: Cell size in meters for discretizing the environment
        """
        self.map_size = map_size
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.grid_width = int(map_size[0] / resolution)
        self.grid_height = int(map_size[1] / resolution)
        
        # Initialize exploration grid
        # -1: obstacle, 0: unexplored, 1+: explored by robot(s)
        self.exploration_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
        
        # Track exploration by each robot separately
        self.robot_exploration_grids = []
        
        # Total explorable area (non-obstacle cells)
        self.total_explorable_area = self.grid_width * self.grid_height
        
    def reset(self, robot_count=1):
        """
        Reset the exploration metrics for a new episode.
        
        Args:
            robot_count: Number of robots in the environment
        """
        # Reset exploration grid
        self.exploration_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
        
        # Reset robot-specific grids
        self.robot_exploration_grids = []
        for _ in range(robot_count):
            self.robot_exploration_grids.append(
                np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
            )
    
    def set_obstacles(self, obstacles):
        """
        Set obstacle cells in the exploration grid.
        
        Args:
            obstacles: List of (x, y) coordinates representing obstacle locations
                      or a 2D numpy array representing the obstacle grid
        """
        if isinstance(obstacles, list):
            for x, y in obstacles:
                # Convert world coordinates to grid coordinates
                grid_x, grid_y = self._world_to_grid(x, y)
                
                # Mark as obstacle if within grid
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    self.exploration_grid[grid_x, grid_y] = -1
        elif isinstance(obstacles, np.ndarray):
            # If obstacles is already a grid, just copy it
            if obstacles.shape == self.exploration_grid.shape:
                # Mark obstacles as -1
                self.exploration_grid = np.where(obstacles > 0, -1, self.exploration_grid)
        
        # Recalculate total explorable area
        self.total_explorable_area = np.sum(self.exploration_grid >= 0)
    
    def update_exploration(self, robot_positions, sensor_ranges):
        """
        Update the exploration grid based on robot positions and sensor ranges.
        
        Args:
            robot_positions: List of (x, y) positions of each robot
            sensor_ranges: List of sensor range values for each robot (or single value for all)
        
        Returns:
            newly_explored: Number of newly explored cells in this update
        """
        # Ensure sensor_ranges is a list
        if not isinstance(sensor_ranges, list):
            sensor_ranges = [sensor_ranges] * len(robot_positions)
        
        # Track newly explored cells
        newly_explored = 0
        
        # Update for each robot
        for i, (pos, sensor_range) in enumerate(zip(robot_positions, sensor_ranges)):
            if i >= len(self.robot_exploration_grids):
                # Add new robot grid if needed
                self.robot_exploration_grids.append(
                    np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
                )
            
            # Get robot's grid position
            robot_x, robot_y = self._world_to_grid(pos[0], pos[1])
            
            # Skip if robot is outside grid
            if not (0 <= robot_x < self.grid_width and 0 <= robot_y < self.grid_height):
                continue
            
            # Calculate grid cells within sensor range
            range_cells = int(sensor_range / self.resolution)
            
            # Simple circle approximation for now
            # For more accurate representation, implement ray-casting
            for dx in range(-range_cells, range_cells + 1):
                for dy in range(-range_cells, range_cells + 1):
                    # Check if cell is within sensor range circle
                    if dx*dx + dy*dy <= range_cells*range_cells:
                        grid_x, grid_y = robot_x + dx, robot_y + dy
                        
                        # Check if cell is within grid boundaries
                        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                            # Don't count obstacle cells
                            if self.exploration_grid[grid_x, grid_y] == -1:
                                continue
                            
                            # Check if this is a newly explored cell for this robot
                            if self.robot_exploration_grids[i][grid_x, grid_y] == 0:
                                self.robot_exploration_grids[i][grid_x, grid_y] = 1
                                
                                # Check if this is also newly explored globally
                                if self.exploration_grid[grid_x, grid_y] == 0:
                                    newly_explored += 1
                                
                                # Increment global exploration count
                                self.exploration_grid[grid_x, grid_y] += 1
        
        return newly_explored
    
    def get_exploration_coverage(self):
        """
        Calculate the percentage of the environment that has been explored.
        
        Returns:
            float: Percentage of explorable area that has been explored
        """
        if self.total_explorable_area <= 0:
            return 0.0
        
        # Count cells that have been explored (value > 0)
        explored_cells = np.sum(self.exploration_grid > 0)
        
        # Calculate percentage of explorable area
        return explored_cells / self.total_explorable_area
    
    def get_exploration_overlap(self):
        """
        Calculate the amount of overlap in exploration between robots.
        
        Returns:
            float: Percentage of exploration effort that was redundant
        """
        # Count total exploration effort (sum of all robot exploration grids)
        total_effort = sum(np.sum(grid > 0) for grid in self.robot_exploration_grids)
        
        if total_effort <= 0:
            return 0.0
        
        # Count unique cells explored (exploration_grid > 0)
        unique_explored = np.sum(self.exploration_grid > 0)
        
        # Calculate overlap
        overlap = (total_effort - unique_explored) / total_effort if total_effort > 0 else 0
        
        return overlap
    
    def get_robot_exploration_stats(self):
        """
        Get exploration statistics for each robot.
        
        Returns:
            list: Dictionary of stats for each robot
        """
        stats = []
        
        for i, grid in enumerate(self.robot_exploration_grids):
            explored = np.sum(grid > 0)
            percentage = explored / self.total_explorable_area if self.total_explorable_area > 0 else 0
            
            stats.append({
                'robot_id': i,
                'explored_cells': int(explored),
                'exploration_percentage': float(percentage)
            })
            
        return stats
    
    def get_unexplored_area(self):
        """
        Get the number of unexplored cells in the environment.
        
        Returns:
            int: Number of unexplored cells
        """
        # Count cells that haven't been explored (value == 0) and aren't obstacles (value != -1)
        unexplored = np.sum(self.exploration_grid == 0)
        return int(unexplored)
    
    def _world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            x, y: World coordinates
            
        Returns:
            tuple: Grid coordinates (grid_x, grid_y)
        """
        # Offset to handle negative coordinates
        offset_x = self.map_size[0] / 2
        offset_y = self.map_size[1] / 2
        
        # Convert to grid coordinates
        grid_x = int((x + offset_x) / self.resolution)
        grid_y = int((y + offset_y) / self.resolution)
        
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            grid_x, grid_y: Grid coordinates
            
        Returns:
            tuple: World coordinates (x, y)
        """
        # Offset to handle negative coordinates
        offset_x = self.map_size[0] / 2
        offset_y = self.map_size[1] / 2
        
        # Convert to world coordinates (center of cell)
        x = grid_x * self.resolution - offset_x + (self.resolution / 2)
        y = grid_y * self.resolution - offset_y + (self.resolution / 2)
        
        return x, y 