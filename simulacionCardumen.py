import numpy as np
import math
import random
import pygame
import sys

# Constantes de configuración
EMPTY = 0
FISH = 1
PREDATOR = 2
OBSTACLE = 3

# Direcciones (8 posibles)
DIRECTIONS = [
    (0, -1),   # N
    (1, -1),   # NE
    (1, 0),    # E
    (1, 1),    # SE
    (0, 1),    # S
    (-1, 1),   # SO
    (-1, 0),   # O
    (-1, -1)   # NO
]

SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
FLEE_WEIGHT = 2.0

SEPARATION_RADIUS = 1
ALIGNMENT_RADIUS = 2
COHESION_RADIUS = 2
FLEE_RADIUS = 3

class CellularAutomaton:
    def __init__(self, width, height, num_fish, num_predators, num_obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width, 2), dtype=int)  # su [tipo, dirección]
        self.initialize_random(num_fish, num_predators, num_obstacles)
        
    def initialize_random(self, num_fish, num_predators, num_obstacles):
        # Inicio de lso peces sapos
        for _ in range(num_fish):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            while self.grid[y, x, 0] != EMPTY:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            direction = random.randint(0, 7)
            self.grid[y, x] = [FISH, direction]
        
        # Inicializar depredadores
        for _ in range(num_predators):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            while self.grid[y, x, 0] != EMPTY:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y, x] = [PREDATOR, -1]
        
        # Inicializar obstáculos
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            while self.grid[y, x, 0] != EMPTY:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y, x] = [OBSTACLE, -1]
    
    def get_neighbors(self, x, y, radius):
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                cell_type = self.grid[ny, nx, 0]
                direction = self.grid[ny, nx, 1]
                neighbors.append(((nx, ny), cell_type, direction))
        return neighbors
    
    def calculate_separation(self, x, y):
        vector = [0.0, 0.0]
        count = 0
        
        for (nx, ny), cell_type, _ in self.get_neighbors(x, y, SEPARATION_RADIUS):
            if cell_type in [FISH, PREDATOR, OBSTACLE]:
                
                dx = (nx - x)
                dy = (ny - y)
                dist = max(0.1, math.sqrt(dx**2 + dy**2))
                vector[0] -= dx / dist
                vector[1] -= dy / dist
                count += 1
        
        if count > 0:
            vector[0] /= count
            vector[1] /= count
            
        return vector
    
    def calculate_alignment(self, x, y):
        vector = [0.0, 0.0]
        count = 0
        
        for _, cell_type, direction in self.get_neighbors(x, y, ALIGNMENT_RADIUS):
            if cell_type == FISH:
                dx, dy = DIRECTIONS[direction]
                vector[0] += dx
                vector[1] += dy
                count += 1
        
        if count > 0:
            # Normalizar
            magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
            if magnitude > 0:
                vector[0] /= magnitude
                vector[1] /= magnitude
        
        return vector
    
    def calculate_cohesion(self, x, y):
        center = [0.0, 0.0]
        count = 0
        
        for (nx, ny), cell_type, _ in self.get_neighbors(x, y, COHESION_RADIUS):
            if cell_type == FISH:
                center[0] += nx
                center[1] += ny
                count += 1
        
        if count > 0:
            center[0] /= count
            center[1] /= count
            # Vector hacia el centro
            vector = [center[0] - x, center[1] - y]
            # Normalizar
            magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
            if magnitude > 0:
                vector[0] /= magnitude
                vector[1] /= magnitude
            return vector
        
        return [0.0, 0.0]
    
    def calculate_flee(self, x, y):
        vector = [0.0, 0.0]
        
        for (nx, ny), cell_type, _ in self.get_neighbors(x, y, FLEE_RADIUS):
            if cell_type == PREDATOR:
                # Vector de huida
                dx = (nx - x)
                dy = (ny - y)
                dist = max(1.0, math.sqrt(dx**2 + dy**2))
                vector[0] -= dx / dist
                vector[1] -= dy / dist
        
        return vector
    
    def calculate_new_direction(self, x, y):
        if self.grid[y, x, 0] != FISH:
            return -1
        
        # Calcular vectores de comportamiento
        sep_vec = self.calculate_separation(x, y)
        ali_vec = self.calculate_alignment(x, y)
        coh_vec = self.calculate_cohesion(x, y)
        flee_vec = self.calculate_flee(x, y)
        
        # Combinar vectores con pesos
        total_vec = [
            SEPARATION_WEIGHT * sep_vec[0] + 
            ALIGNMENT_WEIGHT * ali_vec[0] + 
            COHESION_WEIGHT * coh_vec[0] + 
            FLEE_WEIGHT * flee_vec[0],
            
            SEPARATION_WEIGHT * sep_vec[1] + 
            ALIGNMENT_WEIGHT * ali_vec[1] + 
            COHESION_WEIGHT * coh_vec[1] + 
            FLEE_WEIGHT * flee_vec[1]
        ]
        
        # Convertir vector a dirección
        if total_vec[0] == 0 and total_vec[1] == 0:
            return self.grid[y, x, 1]  # Mantener dirección actual
        
        angle = math.atan2(total_vec[1], total_vec[0])
        sector = int(round(angle / (2 * math.pi / 8)) % 8)
        return sector
    
    def update(self):
        # Paso 1: Calcular nuevas direcciones
        new_directions = np.full((self.height, self.width), -1)
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x, 0] == FISH:
                    new_directions[y, x] = self.calculate_new_direction(x, y)
        
        # Paso 2: Crear nueva grilla y copiar elementos estáticos
        new_grid = np.zeros((self.height, self.width, 2), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x, 0] in [PREDATOR, OBSTACLE]:
                    new_grid[y, x] = self.grid[y, x]
        
        # Paso 3: Mover peces en orden aleatorio
        fish_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x, 0] == FISH:
                    fish_positions.append((x, y))
        
        random.shuffle(fish_positions)
        
        for x, y in fish_positions:
            new_dir = new_directions[y, x]
            dx, dy = DIRECTIONS[new_dir]
            new_x, new_y = (x + dx) % self.width, (y + dy) % self.height
            
            # Intentar mover a la dirección deseada
            if new_grid[new_y, new_x, 0] == EMPTY:
                new_grid[new_y, new_x] = [FISH, new_dir]
            else:
                # Buscar dirección alternativa
                moved = False
                for offset in [1, -1, 2, -2, 3, -3, 4, -4]:
                    alt_dir = (new_dir + offset) % 8
                    dx, dy = DIRECTIONS[alt_dir]
                    alt_x, alt_y = (x + dx) % self.width, (y + dy) % self.height
                    
                    if new_grid[alt_y, alt_x, 0] == EMPTY:
                        new_grid[alt_y, alt_x] = [FISH, alt_dir]
                        moved = True
                        break
                
                # Si no se pudo mover, permanece en su posición con nueva dirección
                if not moved:
                    if new_grid[y, x, 0] == EMPTY:
                        new_grid[y, x] = [FISH, new_dir]
        
        self.grid = new_grid

# Configuración de Pygame para visualización
class SimulationVisualizer:
    def __init__(self, automaton, cell_size=10):
        self.automaton = automaton
        self.cell_size = cell_size
        self.width = automaton.width * cell_size
        self.height = automaton.height * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Simulación de Cardumen con Depredadores y Obstáculos")
        self.clock = pygame.time.Clock()
        
        # Colores
        self.colors = {
            EMPTY: (0, 0, 50),         # Azul marino oscuro
            FISH: (0, 255, 255),       # Cian
            PREDATOR: (255, 0, 0),     # si es Rojo
            OBSTACLE: (100, 100, 100)  # Gris
        }
    
    def draw_grid(self):
        for y in range(self.automaton.height):
            for x in range(self.automaton.width):
                cell_type = self.automaton.grid[y, x, 0]
                color = self.colors[cell_type]
                
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                
                # Dibujar dirección de los peces
                if cell_type == FISH:
                    direction = self.automaton.grid[y, x, 1]
                    dx, dy = DIRECTIONS[direction]
                    center_x = x * self.cell_size + self.cell_size // 2
                    center_y = y * self.cell_size + self.cell_size // 2
                    end_x = center_x + dx * (self.cell_size // 2)
                    end_y = center_y + dy * (self.cell_size // 2)
                    pygame.draw.line(self.screen, (0, 0, 0), 
                                    (center_x, center_y), 
                                    (end_x, end_y), 2)
    
    def run(self, fps=10):
        running = True
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_q:
                        running = False
            
            if not paused:
                self.automaton.update()
            
            self.screen.fill((0, 0, 0))
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(fps)
        
        pygame.quit()

# Parámetros de la simulación
GRID_WIDTH = 80
GRID_HEIGHT = 60
NUM_FISH = 100
NUM_PREDATORS = 5
NUM_OBSTACLES = 20

if __name__ == "__main__":
    automaton = CellularAutomaton(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_fish=NUM_FISH,
        num_predators=NUM_PREDATORS,
        num_obstacles=NUM_OBSTACLES
    )
    
    visualizer = SimulationVisualizer(automaton, cell_size=10)
    visualizer.run(fps=10)