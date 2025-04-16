import numpy as np
import pygame
import random
from PIL import Image
import io
import os
from heapq import heappush, heappop

# Constants
GRID_SIZE = 10
EMPTY = 0
FIRE = 2
AGENT = 3
OBSTACLE = -1
WATER_TILE = 4
DEHYDRATED_AGENT = 5
HOSE_AGENT = 6

FIRE_SPREAD_PROB = 0.35
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
FPS = 5
CELL_SIZE = 100
SCREEN_SIZE = GRID_SIZE * CELL_SIZE  # 1000x1000

# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Add fire locations
fires = [(7, 2), (3, 7)]
for x, y in fires:
    grid[x, y] = FIRE

# Water Location
for i in range(0, GRID_SIZE):  
    grid[0, i] = WATER_TILE

# Agents
agents = [(0, 0), (9, 9), (0, 9), (9, 0), (6, 7),(5,4)]
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

hose_agent = agents[0]  

num_obstacles = 10  # Reduced from 15 to ease navigation
obstacles = set()
cell_size = CELL_SIZE

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Firefighting Simulation (Q-learning + MAPF + Chain)")
clock = pygame.time.Clock()

# Load image with Pygame (for icons)
def load_image(path):
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (cell_size, cell_size))
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)

# Load background image
def load_background_image(path):
    try:
        img = pygame.image.load(path).convert()
        img = pygame.transform.scale(img, (SCREEN_SIZE, SCREEN_SIZE))
        return img
    except Exception as e:
        print(f"Error loading background {path}: {e}")
        # Fallback: solid gray background
        surf = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
        surf.fill((100, 100, 100))
        return surf

agent_img = load_image("assets/bot.png")
agent_dehydrated_img = load_image("assets/bot2.png")
obstacles_img = load_image("assets/Obstacle.png")
agent_hose_img = load_image("assets/bot3.png")
water_tile_img = load_image("assets/water_tank.png")
background_img = load_background_image("assets/Grid_dark.png")

# Function to load GIF frames for Pygame
def load_gif_frames(gif_path, cell_size):
    try:
        gif = Image.open(gif_path)
        frames = []
        while True:
            frame = gif.convert("RGBA")
            frame = frame.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            frame_data = frame.tobytes()
            pygame_frame = pygame.image.fromstring(frame_data, frame.size, "RGBA").convert_alpha()
            frames.append(pygame_frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    except Exception as e:
        print(f"Error loading GIF {gif_path}: {e}")
        return [pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)]
    return frames

# Load the GIF frames
fire_gif_frames = load_gif_frames("assets/fire.gif", cell_size)
fire_frame_index = 0

def is_adjacent_to_agent(x, y):
    for ax, ay in agents:
        if abs(ax - x) + abs(ay - y) <= 1:
            return True
    return False

# Scatter obstacles
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
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 3))  # 3 actions: 0 for FIRE, 1 for WATER_TILE, 2 for PASS_WATER
Q_table[:, :, 0] = 20  # Stronger initial bias toward extinguishing fires
learning_rate = 0.2  # Increased for faster learning
discount_factor = 0.9
exploration_rate = 0.8  # Higher initial exploration
exploration_decay = 0.99

# A* Pathfinding with weighted heuristic and obstacle penalty
def heuristic(a, b):
    base_dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
    # Add penalty for proximity to obstacles
    penalty = 0
    for dx, dy in MOVES:
        nx, ny = a[0] + dx, a[1] + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == OBSTACLE:
            penalty += 5
    return base_dist * 1.5 + penalty  # Weighted heuristic to prioritize shorter paths

def a_star(start, goal, grid, occupied):
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and
                grid[neighbor[0], neighbor[1]] not in [OBSTACLE] and
                neighbor not in occupied):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    print(f"No path found from {start} to {goal}")  # Debug
    return []  # No path found

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

def find_nearest_agent(agent_x, agent_y):
    min_dist = float('inf')
    nearest_agent = None
    for other_agent in agents:
        if other_agent != (agent_x, agent_y):  # Exclude self
            dist = abs(agent_x - other_agent[0]) + abs(agent_y - other_agent[1])
            if dist <= 1 and dist < min_dist:  # Adjacent only
                min_dist = dist
                nearest_agent = other_agent
    print(f"Nearest agent to {agent_x}, {agent_y} is {nearest_agent}")  # Debug
    return nearest_agent

def extinguish_fire(x, y):
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx, ny] == FIRE:
                    grid[nx, ny] = EMPTY
                    print(f"Fire extinguished at {nx}, {ny}")  # Debug

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
        print(f"Hose extinguished fire at {x}, {y}, water remaining: {state['water_remaining']}")  # Debug
        return True
    return False

def pass_water(giver_pos, receiver_pos):
    giver_state = agent_states[giver_pos]
    receiver_state = agent_states[receiver_pos]
    if giver_state["water_remaining"] > 1 and receiver_state["water_remaining"] < receiver_state["water_capacity"]:
        water_to_pass = min(2, giver_state["water_remaining"] - 1, receiver_state["water_capacity"] - receiver_state["water_remaining"])
        giver_state["water_remaining"] -= water_to_pass
        receiver_state["water_remaining"] += water_to_pass
        receiver_state["needs_refill"] = False
        print(f"Water passed from {giver_pos} to {receiver_pos}, amounts: {giver_state['water_remaining']} -> {receiver_state['water_remaining']}")  # Debug
        return True
    return False

def move_agents():
    global agents, agent_states, grid, exploration_rate
    new_agents = []
    new_agent_states = {}
    occupied = set()  # Cleared per agent to improve navigation

    for agent in agents:
        x, y = agent
        state = agent_states[agent]
        current_state = (x, y)

        # Choose action using Q-learning with epsilon-greedy
        if random.random() < exploration_rate:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Q_table[x, y])

        if action == 2 and state["water_remaining"] > 1:  # Pass water
            nearest_agent = find_nearest_agent(x, y)
            if nearest_agent is not None and pass_water((x, y), nearest_agent):
                reward = 20 if any(grid[nx, ny] == FIRE for nx, ny in [(x+dx, y+dy) for dx, dy in MOVES] if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) else 5
                next_max = max(Q_table[x, y, 0], Q_table[x, y, 1], Q_table[x, y, 2])
                old_value = Q_table[x, y, action]
                Q_table[x, y, action] = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
                new_agents.append(agent)
                new_agent_states[agent] = state
                continue  # Skip to next agent after passing water

        # Prioritize fire unless dehydrated
        if action == 0 or (action == 2 and (state["water_remaining"] <= 1 or nearest_agent is None)):
            target_type = FIRE if state["water_remaining"] > 0 else WATER_TILE
        elif action == 1:
            target_type = WATER_TILE

        target = find_nearest_target(x, y, target_type)
        if target:
            path = a_star((x, y), target, grid, occupied)
            if path and len(path) > 1:  # Move to next position if path exists
                next_pos = path[1]
                nx, ny = next_pos

                # Check for collisions
                if next_pos not in occupied:
                    reward = -1  # Default movement penalty
                    if grid[nx, ny] == FIRE and state["water_remaining"] > 0:
                        distance_reward = 10 * (len(path) - 1)  # Bonus for shorter paths to fire
                        if agent == hose_agent:
                            if hose_extinguish_fire(nx, ny, state):
                                reward = 50 + distance_reward
                        else:
                            extinguish_fire(nx, ny)
                            reward = 50 + distance_reward
                        state["water_remaining"] -= 1
                        if state["water_remaining"] == 0:
                            state["needs_refill"] = True
                    elif grid[nx, ny] == WATER_TILE and state["needs_refill"]:
                        reward = 10
                        state["water_remaining"] = state["water_capacity"]
                        state["needs_refill"] = False

                    # Q-learning update
                    old_value = Q_table[x, y, action]
                    next_max = max(Q_table[nx, ny, 0], Q_table[nx, ny, 1], Q_table[nx, ny, 2])
                    new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
                    Q_table[x, y, action] = new_value

                    grid[x, y] = WATER_TILE if x == 0 else EMPTY
                    grid[nx, ny] = HOSE_AGENT if agent == hose_agent else (DEHYDRATED_AGENT if state["needs_refill"] else AGENT)
                    new_agents.append((nx, ny))
                    new_agent_states[(nx, ny)] = state
                    occupied.add((nx, ny))  # Reserve the next position
                else:
                    new_agents.append(agent)
                    new_agent_states[agent] = state
            else:
                new_agents.append(agent)
                new_agent_states[agent] = state
        else:
            new_agents.append(agent)
            new_agent_states[agent] = state

    agents = new_agents
    agent_states = new_agent_states
    exploration_rate *= exploration_decay  # Decay exploration rate

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

def show_grid():
    global fire_frame_index
    fire_frame_index = (fire_frame_index + 1) % len(fire_gif_frames)

    # Clear screen and draw background
    screen.fill((0, 0, 0))  # Black fallback
    screen.blit(background_img, (0, 0))

    # Overlay dynamic elements
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            px = y * cell_size  # Pygame: x-axis is horizontal
            py = x * cell_size  # Pygame: y-axis is vertical
            if grid[x, y] == OBSTACLE:
                screen.blit(obstacles_img, (px, py))
            elif grid[x, y] == FIRE:
                fire_frame = fire_gif_frames[fire_frame_index]
                screen.blit(fire_frame, (px, py))
            elif grid[x, y] == WATER_TILE:
                screen.blit(water_tile_img, (px, py))
            elif grid[x, y] == AGENT:
                screen.blit(agent_img, (px, py))
            elif grid[x, y] == DEHYDRATED_AGENT:
                screen.blit(agent_dehydrated_img, (px, py))
            elif grid[x, y] == HOSE_AGENT:
                screen.blit(agent_hose_img, (px, py))

    # Render and display completion message if simulation is complete
    if 'simulation_complete' in globals() and simulation_complete:
        font = pygame.font.SysFont(None, 74)
        text = font.render("All fires extinguished!!", True, (255, 255, 0))  # Yellow text
        text_rect = text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2))
        screen.blit(text, text_rect)

    pygame.display.flip()

def run_simulation():
    global exploration_rate, simulation_complete
    running = True
    step = 0
    simulation_complete = False

    try:
        while running and step < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            print(f"🔥 Step {step+1}")
            # Debug: Print grid state
            print("Grid state:")
            for row in grid:
                print([int(x) for x in row])
            show_grid()
            spread_fire()
            move_agents()
            if not any(grid[x, y] == FIRE for x in range(GRID_SIZE) for y in range(GRID_SIZE)):
                print("🎉 All fires extinguished!")
                simulation_complete = True
                break

            step += 1
            clock.tick(FPS)

        # Keep window open with completion message until closed
        while simulation_complete and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            show_grid()
            clock.tick(FPS)

    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        pygame.quit()

# Run the simulation
try:
    run_simulation()
except Exception as e:
    print(f"Simulation crashed: {e}")
    pygame.quit()