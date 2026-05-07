import random

UP    = (-1,  0)
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)

TURN_LEFT = {
    UP:    LEFT,
    LEFT:  DOWN,
    DOWN:  RIGHT,
    RIGHT: UP,
}
TURN_RIGHT = {
    UP:    RIGHT,
    RIGHT: DOWN,
    DOWN:  LEFT,
    LEFT:  UP,
}

class SnakeEnv:

    BASE_LOOP_LIMIT = 100  

    def __init__(self, grid_size=10, obstacles=True):
        
        self.GRID_SIZE     = grid_size
        self.use_obstacles = obstacles
        a = grid_size // 3
        b = grid_size - a - 1
        self.obstacles = [(a, a), (a, a+1), (b, b), (b, b+1)] if obstacles else []

        self.reset()

    def reset(self):
        
        mid = self.GRID_SIZE // 2
        self.snake = [
            (mid, mid),
            (mid, mid - 1),
            (mid, mid - 2),
        ]
        self.direction        = RIGHT
        self.score            = 0
        self.steps            = 0
        self.steps_since_food = 0
        self.speed_level      = 1   
        
        self._spawn_food()
        return self._get_state()

    def step(self, action):
        
        self.steps += 1
        self.steps_since_food += 1

        if action == 1:
            self.direction = TURN_LEFT[self.direction]
        elif action == 2:
            self.direction = TURN_RIGHT[self.direction]

        head_r, head_c = self.snake[0]
        dr, dc = self.direction
        new_head = (head_r + dr, head_c + dc)

        if self._is_collision(new_head):
            return self._get_state(), -10, True, self._info()

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score            += 1
            self.steps_since_food  = 0
            reward                 = +10
            self._spawn_food()
            self.speed_level = 1 + (self.score // 3)
            
        else:
            self.snake.pop()   
            reward = -0.1

        loop_limit = self.BASE_LOOP_LIMIT * (self.GRID_SIZE // 10)
        if self.steps_since_food > loop_limit:
            return self._get_state(), -10, True, self._info()

        return self._get_state(), reward, False, self._info()

    def _info(self):

        return {
            "score":        self.score,
            "speed_level":  self.speed_level,    
            "snake_length": len(self.snake),
            "steps":        self.steps,
        }

    def _get_state(self):
       
        head = self.snake[0]
        head_r, head_c = head

        straight_pos = self._pos_in_direction(head, self.direction)
        left_pos      = self._pos_in_direction(head, TURN_LEFT[self.direction])
        right_pos     = self._pos_in_direction(head, TURN_RIGHT[self.direction])

        food_r, food_c = self.food

        state = [
            int(self._is_collision(straight_pos)),   
            int(self._is_collision(left_pos)),       
            int(self._is_collision(right_pos)),      

            int(food_c < head_c),   
            int(food_c > head_c),   
            int(food_r < head_r),   
            int(food_r > head_r),   

            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            int(self.direction == UP),
            int(self.direction == DOWN),
        ]

        return state  

    def _is_collision(self, pos):

        r, c = pos

        if r < 0 or r >= self.GRID_SIZE or c < 0 or c >= self.GRID_SIZE:
            return True

        if pos in self.snake[1:]:
            return True

        if pos in self.obstacles:
            return True

        return False

    def _spawn_food(self):

        all_cells = [
            (r, c)
            for r in range(self.GRID_SIZE)
            for c in range(self.GRID_SIZE)
        ]
        occupied = set(self.snake) | set(self.obstacles)
        free_cells = [cell for cell in all_cells if cell not in occupied]

        if free_cells:
            self.food = random.choice(free_cells)
        else:
            self.food = None

    def _pos_in_direction(self, pos, direction):

        r, c = pos
        dr, dc = direction
        return (r + dr, c + dc)

    def render(self):

        grid = [['.' for _ in range(self.GRID_SIZE)]
                for _ in range(self.GRID_SIZE)]

        for (r, c) in self.obstacles:
            grid[r][c] = '#'
        for (r, c) in self.snake[1:]:
            grid[r][c] = 'o'
        hr, hc = self.snake[0]
        grid[hr][hc] = 'H'
        if self.food:
            fr, fc = self.food
            grid[fr][fc] = '*'

        dir_symbol = {RIGHT: '→', LEFT: '←', UP: '↑', DOWN: '↓'}
        print(f"\nGrid:{self.GRID_SIZE}x{self.GRID_SIZE} | "
              f"Score:{self.score} | Steps:{self.steps} | "
              f"Speed Level:{self.speed_level} | "       
              f"Length:{len(self.snake)} | "
              f"Dir:{dir_symbol.get(self.direction,'?')}")
        print("+" + "-" * self.GRID_SIZE + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * self.GRID_SIZE + "+")

if __name__ == "__main__":
    print("=" * 52)
    print("  TEST: Snake Environment — 10x10 with obstacles")
    print("=" * 52)

    env   = SnakeEnv(grid_size=10, obstacles=True)
    state = env.reset()
    print("Initial state (11 elements):", state)
    print("State length:", len(state))
    env.render()

    for i in range(10):
        action = random.randint(0, 2)
        state, reward, done, info = env.step(action)
        print(f"Step {i+1:2d} | Action:{action} | Reward:{reward:+.1f} | "
              f"Done:{done} | Speed Level:{info['speed_level']}")
        if done:
            print("  → Episode ended, resetting...")
            env.reset()

    env.render()