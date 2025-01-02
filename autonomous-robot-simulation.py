import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
import heapq

# **1. Define the Workspace**
# Smaller grid dimensions
arena_size = (10, 10)
# 0: Free space, 1: Obstacle
arena = np.zeros(arena_size)

# Define initial positions for dynamic obstacles
num_obstacles = 5  # Fewer obstacles
np.random.seed(42)  # Fixed randomness for reproducibility
obstacle_positions = set()

while len(obstacle_positions) < num_obstacles:
    x, y = np.random.randint(0, arena_size[0]), np.random.randint(0, arena_size[1])
    if (x, y) != (0, 0) and (x, y) != (arena_size[0]-1, arena_size[1]-1):
        obstacle_positions.add((x, y))

for pos in obstacle_positions:
    arena[pos] = 1

# **2. Define Start and Goal Points**
start = (0, 0)
goal = (arena_size[0]-1, arena_size[1]-1)
arena[start] = 0  # Ensure start point is not an obstacle
arena[goal] = 0   # Ensure goal point is not an obstacle

# **3. Path Planning with A* Algorithm**
def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(arena, start, goal):
    """Find the shortest path from start to goal using the A* algorithm."""
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < arena_size[0] and 0 <= neighbor[1] < arena_size[1]:
                if arena[neighbor] == 1:
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    priority = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, tentative_g_score, neighbor))
                    came_from[neighbor] = current

    return []  # Return an empty path if the goal is unreachable

# **4. Dynamic Obstacle Movement**
def move_obstacles():
    """Randomly move obstacles."""
    global obstacle_positions, arena
    new_positions = set()
    for pos in obstacle_positions:
        x, y = pos
        arena[x, y] = 0  # Clear the old position

        # Choose a random direction
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        new_x, new_y = x + dx, y + dy

        # Check if the new position is valid
        if 0 <= new_x < arena_size[0] and 0 <= new_y < arena_size[1]:
            if (new_x, new_y) not in obstacle_positions and (new_x, new_y) != start and (new_x, new_y) != goal:
                new_positions.add((new_x, new_y))
                arena[new_x, new_y] = 1  # Mark the new position
            else:
                new_positions.add(pos)  # Stay in the same place
        else:
            new_positions.add(pos)  # Stay in the same place

    obstacle_positions = new_positions

# **5. Simulation and Path Planning**
robot_path = astar(arena, start, goal)

if not robot_path:
    print("The robot cannot reach the goal! No path found.")
else:
    print(f"The robot reached the goal in {len(robot_path)} steps.")

# **6. Animation**
def animate(i):
    global robot_path
    if i % 5 == 0:  # Move obstacles every 5 frames
        move_obstacles()
        robot_path = astar(arena, start, goal)  # Recalculate the path

    ax.clear()
    ax.imshow(arena, cmap="Greys", origin="upper")
    # Draw the robot's path as blue dots
    if i < len(robot_path):
        ax.scatter([p[1] for p in robot_path[:i+1]], [p[0] for p in robot_path[:i+1]], c='blue', label='Robot Path', s=50)
    ax.scatter(start[1], start[0], c='green', label='Start', s=100)  # Start point
    ax.scatter(goal[1], goal[0], c='red', label='Goal', s=100)      # Goal point
    ax.legend()
    ax.set_title("Autonomous Robot Simulation - Dynamic Obstacles")
    ax.set_xticks(range(arena_size[1]))
    ax.set_yticks(range(arena_size[0]))
    ax.set_xlim(-0.5, arena_size[1] - 0.5)
    ax.set_ylim(-0.5, arena_size[0] - 0.5)
    ax.grid(True, which='both', color='black', linestyle='--', linewidth=0.5)

fig, ax = plt.subplots(figsize=(8, 8))
ani = FuncAnimation(fig, animate, frames=50, interval=200, repeat=False)

# Save animation as a GIF
output_file = "autonomous_robot_simulation.gif"
writer = PillowWriter(fps=5)
ani.save(output_file, writer=writer)

print(f"GIF saved successfully: {output_file}")
