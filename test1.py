import numpy as np
import cv2
import random
from PIL import Image

# Constants
GRID_SIZE = 10
EMPTY = 0
FIRE = 2
AGENT = 3
OBSTACLE = -1
WATER_TILE = 4
DEHYDRATED_AGENT = 5
HOSE_AGENT = 6

FIRE_SPREAD_PROB = 0.55
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right


# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
# Add water buckets to the left and right sides (middle row)
grid[5, 0] = WATER_TILE  # Left side
grid[5, GRID_SIZE - 1] = WATER_TILE  # Right side

# Add water buckets at the bottom row
grid[GRID_SIZE - 1, 3] = WATER_TILE
grid[GRID_SIZE - 1, 6] = WATER_TILE

# Add fire locations
fires = [(7, 2), (9, 7), (8,6)]
for x, y in fires:
    grid[x, y] = FIRE

# Add water refill tiles (alternating blue tiles in top row)
for i in range(GRID_SIZE):
    grid[0, i] = WATER_TILE

# Add agents
agents = [(0, 0), (9, 9), (0, 9), (9, 0), (6, 7)]
agent_states = {
    agent: {
        "water_capacity": 5,
        "water_remaining": 3,
        "needs_refill": False,
        "last_fire_extinguished": None,
        "known_obstacles": set()
    } for agent in agents
}
for x, y in agents:
    grid[x, y] = AGENT

hose_agent = agents[0]  # First agent has a hose

# Randomly generate obstacles
num_obstacles = 10
obstacles = set()
cell_size = 100  # Image display cell size

# Load image with transparency
def load_image(path, scale=1.0):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.resize(img, (cell_size, cell_size))

agent_img = load_image("assets\\bot.png", scale=0.85)
agent_dehydrated_img = load_image("assets\\bot2.png")
obstacles_img = load_image("assets\\Obstacle.png", scale=1.0)
agent_hose_img = load_image("assets\\bot3.png")
water_tile_img = load_image("assets\\water_tank.png", scale=2.0)


def is_adjacent_to_agent(x, y):
    for ax, ay in agents:
        if abs(ax - x) + abs(ay - y) <= 1:
            return True
    return False

# Scatter obstacles across grid
region_size = max(1, GRID_SIZE // int(num_obstacles**0.5))
for i in range(0, GRID_SIZE, region_size):
    for j in range(0, GRID_SIZE, region_size):
        if len(obstacles) >= num_obstacles:
            break
        attempts = 0
        while attempts < 10:
            rx = random.randint(i, min(i + region_size - 1, GRID_SIZE - 1))
            ry = random.randint(j, min(j + region_size - 1, GRID_SIZE - 1))
            if rx == 1 or grid[rx, ry] != EMPTY or is_adjacent_to_agent(rx, ry):
                attempts += 1
                continue
            obstacles.add((rx, ry))
            grid[rx, ry] = OBSTACLE
            break

# Q-learning parameters
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(MOVES)))
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.99

def spread_fire():
    new_fires = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == FIRE:
                for dx, dy in MOVES:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        if grid[nx, ny] == EMPTY and np.random.rand() < FIRE_SPREAD_PROB:
                            new_fires.append((nx, ny))
    for x, y in new_fires:
        grid[x, y] = FIRE

def find_nearest_target(agent_x, agent_y, target_type):
    min_dist = float('inf')
    nearest_target = None
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == target_type:
                dist = abs(agent_x - x) + abs(agent_y - y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_target = (x, y)
    return nearest_target

def extinguish_fire(x, y):
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx, ny] == FIRE:
                    grid[nx, ny] = EMPTY


def hose_extinguish_fire(x, y, state):
    if state["water_remaining"] == 0:
        state["needs_refill"] = True
        return False
    extinguished = False
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx, ny] == FIRE:
                    grid[nx, ny] = EMPTY
                    extinguished = True
    if extinguished:
        state["water_remaining"] -= 1
        if state["water_remaining"] == 0:
            state["needs_refill"] = True
        return True
    return False

def move_agents():
    global agents, agent_states
    new_agents = []
    new_agent_states = {}

    for agent in agents:
        x, y = agent
        state = agent_states[agent]

        if agent == hose_agent and not state["needs_refill"]:
            if hose_extinguish_fire(x, y, state):
                new_agents.append(agent)
                new_agent_states[agent] = state
                continue

        target = find_nearest_target(x, y, WATER_TILE if state["needs_refill"] else FIRE)
        if target:
            tx, ty = target
            best_move = None
            best_distance = float('inf')
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if (nx, ny) not in state["known_obstacles"]:
                        if grid[nx, ny] in [EMPTY, FIRE, WATER_TILE]:
                            dist = abs(nx - tx) + abs(ny - ty)
                            if dist < best_distance:
                                best_distance = dist
                                best_move = (nx, ny)
                        elif grid[nx, ny] == OBSTACLE:
                            state["known_obstacles"].add((nx, ny))

            if best_move:
                nx, ny = best_move
                reward = -1
                if grid[nx, ny] == FIRE and not state["needs_refill"]:
                    reward = 50
                    extinguish_fire(nx, ny)
                    state["water_remaining"] -= 1
                    if state["water_remaining"] == 0:
                        state["needs_refill"] = True
                elif grid[nx, ny] == WATER_TILE and state["needs_refill"]:
                    reward = 10
                    state["water_remaining"] = state["water_capacity"]
                    state["needs_refill"] = False

                max_future_q = np.max(Q_table[nx, ny])
                move_idx = MOVES.index((nx - x, ny - y))
                current_q = Q_table[x, y, move_idx]
                Q_table[x, y, move_idx] = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)

                grid[x, y] = WATER_TILE if x == 0 else EMPTY
                grid[nx, ny] = HOSE_AGENT if agent == hose_agent else (DEHYDRATED_AGENT if state["needs_refill"] else AGENT)

                new_agents.append((nx, ny))
                new_agent_states[(nx, ny)] = state
            else:
                new_agents.append(agent)
                new_agent_states[agent] = state
        else:
            new_agents.append(agent)
            new_agent_states[agent] = state

    agents = new_agents
    agent_states = new_agent_states


def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:
        for i in range(h):
            for j in range(w):
                alpha = overlay[i, j, 3] / 255.0
                if alpha > 0:
                    for c in range(3):
                        background[y + i, x + j, c] = (
                            alpha * overlay[i, j, c] + (1 - alpha) * background[y + i, x + j, c]
                        )


# Function to load frames from a GIF
def load_gif_frames(gif_path, cell_size):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = np.array(gif.convert("RGBA")) 
            frame = cv2.resize(frame, (cell_size, cell_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)  # 🔥 This is the key fix
            frames.append(frame)
            gif.seek(gif.tell() + 1)  
    except EOFError:
        pass  
    return frames


# Load the GIF frames
fire_gif_frames = load_gif_frames("assets/fire.gif", cell_size)

# Initialize global fire_frame_index
fire_frame_index = 0
fire_frame_index = (fire_frame_index + 2) % len(fire_gif_frames)  # Skip every other frame




def show_grid():
    global fire_frame_index
    fire_frame_index = (fire_frame_index + 1) % len(fire_gif_frames)

    cell_size = 100
    expected_width = GRID_SIZE * cell_size  # 800
    expected_height = GRID_SIZE * cell_size  # 800
    
    # Load the full background image (should be exactly grid size)
    background = cv2.imread('assets/Grid.jpg')
    background.shape  # should be (800, 800, 3) or (800, 800, 4)

    if background is None:
        raise FileNotFoundError("Could not load 'assets/tiles2.jpg'. Check the file path and existence.")

    # Check if background image matches expected dimensions
    expected_height = GRID_SIZE * cell_size
    expected_width = GRID_SIZE * cell_size
    if background.shape[1] != expected_width or background.shape[0] != expected_height:
        background = cv2.resize(background, (expected_width, expected_height))
    if background.shape[0] != expected_height or background.shape[1] != expected_width:
        raise ValueError(f"Image size mismatch. Expected {expected_width}x{expected_height}, got {background.shape[1]}x{background.shape[0]}")

    # Create canvas as a copy
    img = background.copy()

    # Overlay dynamic elements (fire, agents, etc.)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            py = x * cell_size
            px = y * cell_size

            if grid[x, y] == OBSTACLE:
                overlay_image(img, obstacles_img, px, py)
            elif grid[x, y] == FIRE:
                fire_frame = fire_gif_frames[fire_frame_index]
                overlay_image(img, fire_frame, px, py)
            elif grid[x, y] == WATER_TILE:
                overlay_image(img, water_tile_img, px, py)
            elif grid[x, y] == AGENT:
                overlay_image(img, agent_img, px, py)
            elif grid[x, y] == DEHYDRATED_AGENT:
                overlay_image(img, agent_dehydrated_img, px, py)
            elif grid[x, y] == HOSE_AGENT:
                overlay_image(img, agent_hose_img, px, py)

    cv2.imshow("Firefighting Simulation", img)
    cv2.waitKey(1)


def run_simulation():
    global exploration_rate
    for step in range(100):
        print(f"🔥 Step {step+1}")
        show_grid()
        spread_fire()
        move_agents()
        exploration_rate *= exploration_decay
        if not any(grid[x, y] == FIRE for x in range(GRID_SIZE) for y in range(GRID_SIZE)):
            print("🎉 All fires extinguished!")
            break
    cv2.destroyAllWindows()

# Run the simulation
run_simulation()
print("Barrier image shape:", obstacles_img.shape)
