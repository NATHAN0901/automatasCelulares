import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

# Constantes de configuración
TAMAÑO = 25  # Tamaño del espacio 3D
PASOS = 50   # Número de pasos de simulación

# Estados celulares
EMPTY = 0    # Espacio vacío
FISH = 1     # Pez
PREDATOR = 2 # Depredador
OBSTACLE = 3 # Obstáculo

# Parámetros de comportamiento
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
FLEE_WEIGHT = 2.0

SEPARATION_RADIUS = 1
ALIGNMENT_RADIUS = 2
COHESION_RADIUS = 2
FLEE_RADIUS = 3

# Inicializar grid 3D
grid = np.zeros((TAMAÑO, TAMAÑO, TAMAÑO), dtype=np.uint8)

# Posiciones y direcciones de los peces
fish_positions = []
fish_directions = []

# Posiciones de depredadores y obstáculos
predator_positions = []
obstacle_positions = []

# Inicializar entidades aleatoriamente
def inicializar_entidades(num_fish, num_predators, num_obstacles):
    global fish_positions, fish_directions, predator_positions, obstacle_positions
    
    # Inicializar peces
    for _ in range(num_fish):
        pos = (random.randint(0, TAMAÑO-1), 
               random.randint(0, TAMAÑO-1), 
               random.randint(0, TAMAÑO-1))
        while grid[pos] != EMPTY:
            pos = (random.randint(0, TAMAÑO-1), 
                   random.randint(0, TAMAÑO-1), 
                   random.randint(0, TAMAÑO-1))
        grid[pos] = FISH
        fish_positions.append(pos)
        # Dirección inicial aleatoria
        fish_directions.append((
            random.choice([-1, 0, 1]),
            random.choice([-1, 0, 1]),
            random.choice([-1, 0, 1])
        ))
    
    # Inicializar depredadores
    for _ in range(num_predators):
        pos = (random.randint(0, TAMAÑO-1), 
               random.randint(0, TAMAÑO-1), 
               random.randint(0, TAMAÑO-1))
        while grid[pos] != EMPTY:
            pos = (random.randint(0, TAMAÑO-1), 
                   random.randint(0, TAMAÑO-1), 
                   random.randint(0, TAMAÑO-1))
        grid[pos] = PREDATOR
        predator_positions.append(pos)
    
    # Inicializar obstáculos
    for _ in range(num_obstacles):
        pos = (random.randint(0, TAMAÑO-1), 
               random.randint(0, TAMAÑO-1), 
               random.randint(0, TAMAÑO-1))
        while grid[pos] != EMPTY:
            pos = (random.randint(0, TAMAÑO-1), 
                   random.randint(0, TAMAÑO-1), 
                   random.randint(0, TAMAÑO-1))
        grid[pos] = OBSTACLE
        obstacle_positions.append(pos)

# Obtener vecinos en 3D
def obtener_vecinos_3d(pos, radius):
    x, y, z = pos
    vecinos = []
    for dz in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = (x + dx) % TAMAÑO, (y + dy) % TAMAÑO, (z + dz) % TAMAÑO
                vecinos.append((nx, ny, nz))
    return vecinos

# Calcular vectores de comportamiento
def calcular_separacion(pos):
    vector = [0.0, 0.0, 0.0]
    count = 0
    
    for vecino in obtener_vecinos_3d(pos, SEPARATION_RADIUS):
        if grid[vecino] in [FISH, PREDATOR, OBSTACLE]:
            dx = vecino[0] - pos[0]
            dy = vecino[1] - pos[1]
            dz = vecino[2] - pos[2]
            dist = max(0.1, np.sqrt(dx**2 + dy**2 + dz**2))
            vector[0] -= dx / dist
            vector[1] -= dy / dist
            vector[2] -= dz / dist
            count += 1
    
    if count > 0:
        vector[0] /= count
        vector[1] /= count
        vector[2] /= count
    
    return vector

def calcular_alineacion(pos, idx):
    vector = [0.0, 0.0, 0.0]
    count = 0
    
    for vecino in obtener_vecinos_3d(pos, ALIGNMENT_RADIUS):
        if grid[vecino] == FISH:
            # Encontrar el pez vecino
            if vecino in fish_positions:
                vec_idx = fish_positions.index(vecino)
                dx, dy, dz = fish_directions[vec_idx]
                vector[0] += dx
                vector[1] += dy
                vector[2] += dz
                count += 1
    
    if count > 0:
        magnitude = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        if magnitude > 0:
            vector[0] /= magnitude
            vector[1] /= magnitude
            vector[2] /= magnitude
    
    return vector

def calcular_cohesion(pos):
    center = [0.0, 0.0, 0.0]
    count = 0
    
    for vecino in obtener_vecinos_3d(pos, COHESION_RADIUS):
        if grid[vecino] == FISH:
            center[0] += vecino[0]
            center[1] += vecino[1]
            center[2] += vecino[2]
            count += 1
    
    if count > 0:
        center[0] /= count
        center[1] /= count
        center[2] /= count
        vector = [center[0] - pos[0], center[1] - pos[1], center[2] - pos[2]]
        magnitude = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        if magnitude > 0:
            vector[0] /= magnitude
            vector[1] /= magnitude
            vector[2] /= magnitude
        return vector
    
    return [0.0, 0.0, 0.0]

def calcular_huida(pos):
    vector = [0.0, 0.0, 0.0]
    
    for vecino in obtener_vecinos_3d(pos, FLEE_RADIUS):
        if grid[vecino] == PREDATOR:
            dx = vecino[0] - pos[0]
            dy = vecino[1] - pos[1]
            dz = vecino[2] - pos[2]
            dist = max(1.0, np.sqrt(dx**2 + dy**2 + dz**2))
            vector[0] -= dx / dist
            vector[1] -= dy / dist
            vector[2] -= dz / dist
    
    return vector

# Calcular nueva dirección para un pez
def calcular_nueva_direccion(pos, idx):
    sep_vec = calcular_separacion(pos)
    ali_vec = calcular_alineacion(pos, idx)
    coh_vec = calcular_cohesion(pos)
    flee_vec = calcular_huida(pos)
    
    total_vec = [
        SEPARATION_WEIGHT * sep_vec[0] + 
        ALIGNMENT_WEIGHT * ali_vec[0] + 
        COHESION_WEIGHT * coh_vec[0] + 
        FLEE_WEIGHT * flee_vec[0],
        
        SEPARATION_WEIGHT * sep_vec[1] + 
        ALIGNMENT_WEIGHT * ali_vec[1] + 
        COHESION_WEIGHT * coh_vec[1] + 
        FLEE_WEIGHT * flee_vec[1],
        
        SEPARATION_WEIGHT * sep_vec[2] + 
        ALIGNMENT_WEIGHT * ali_vec[2] + 
        COHESION_WEIGHT * coh_vec[2] + 
        FLEE_WEIGHT * flee_vec[2]
    ]
    
    # Normalizar el vector resultante
    magnitude = np.sqrt(total_vec[0]**2 + total_vec[1]**2 + total_vec[2]**2)
    if magnitude > 0:
        total_vec = [total_vec[0]/magnitude, total_vec[1]/magnitude, total_vec[2]/magnitude]
    
    # Convertir a dirección discreta (aproximar a movimiento en ejes)
    new_dir = [
        1 if total_vec[0] > 0.33 else -1 if total_vec[0] < -0.33 else 0,
        1 if total_vec[1] > 0.33 else -1 if total_vec[1] < -0.33 else 0,
        1 if total_vec[2] > 0.33 else -1 if total_vec[2] < -0.33 else 0
    ]
    
    # Si no hay dirección clara, mantener la anterior
    if new_dir == [0, 0, 0]:
        return fish_directions[idx]
    
    return new_dir

# Mover depredadores de forma aleatoria
def mover_depredadores():
    global predator_positions
    
    new_predator_positions = []
    for pos in predator_positions:
        # Intentar moverse en dirección aleatoria
        dx, dy, dz = random.choice([
            (d[0], d[1], d[2]) for d in [
                (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1),
                (-1,-1,0), (-1,1,0), (1,-1,0), (1,1,0),
                (-1,0,-1), (-1,0,1), (1,0,-1), (1,0,1),
                (0,-1,-1), (0,-1,1), (0,1,-1), (0,1,1)
            ] if d != (0,0,0)
        ])
        
        new_pos = (
            (pos[0] + dx) % TAMAÑO,
            (pos[1] + dy) % TAMAÑO,
            (pos[2] + dz) % TAMAÑO
        )
        
        # Solo moverse si la nueva posición está vacía
        if grid[new_pos] == EMPTY:
            grid[pos] = EMPTY
            grid[new_pos] = PREDATOR
            new_predator_positions.append(new_pos)
        else:
            new_predator_positions.append(pos)
    
    predator_positions = new_predator_positions

# Simular un paso completo
def simular_paso():
    global fish_positions, fish_directions, grid
    
    # Calcular nuevas direcciones para todos los peces
    new_directions = []
    for idx, pos in enumerate(fish_positions):
        new_directions.append(calcular_nueva_direccion(pos, idx))
    
    # Actualizar direcciones
    fish_directions = new_directions
    
    # Crear nueva grid temporal
    new_grid = np.copy(grid)
    for pos in fish_positions:
        new_grid[pos] = EMPTY
    
    # Mover peces
    new_fish_positions = []
    for idx, pos in enumerate(fish_positions):
        dx, dy, dz = fish_directions[idx]
        new_pos = (
            (pos[0] + dx) % TAMAÑO,
            (pos[1] + dy) % TAMAÑO,
            (pos[2] + dz) % TAMAÑO
        )
        
        # Si la nueva posición está vacía, mover
        if new_grid[new_pos] == EMPTY:
            new_grid[new_pos] = FISH
            new_fish_positions.append(new_pos)
        else:
            # Intentar moverse en una dirección alternativa
            moved = False
            directions_to_try = [
                (dx, dy, dz),  # Primero intentar la dirección original
                (dx, dy, 0), (dx, 0, dz), (0, dy, dz),
                (dx, 0, 0), (0, dy, 0), (0, 0, dz),
                (-dx, dy, dz), (dx, -dy, dz), (dx, dy, -dz)
            ]
            
            for d in directions_to_try:
                alt_pos = (
                    (pos[0] + d[0]) % TAMAÑO,
                    (pos[1] + d[1]) % TAMAÑO,
                    (pos[2] + d[2]) % TAMAÑO
                )
                if new_grid[alt_pos] == EMPTY:
                    new_grid[alt_pos] = FISH
                    new_fish_positions.append(alt_pos)
                    moved = True
                    break
            
            # Si no se pudo mover, permanecer en la posición actual
            if not moved:
                new_grid[pos] = FISH
                new_fish_positions.append(pos)
    
    # Actualizar depredadores
    mover_depredadores()
    
    # Actualizar estado global
    fish_positions = new_fish_positions
    grid = new_grid
    
    # Mantener obstáculos
    for pos in obstacle_positions:
        grid[pos] = OBSTACLE

# Visualización 3D
def visualizar_3d(paso):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Preparar datos para visualización
    peces = [[], [], []]
    depredadores = [[], [], []]
    obstaculos = [[], [], []]
    
    # Recopilar coordenadas
    for z in range(TAMAÑO):
        for y in range(TAMAÑO):
            for x in range(TAMAÑO):
                if grid[x, y, z] == FISH:
                    peces[0].append(x)
                    peces[1].append(y)
                    peces[2].append(z)
                elif grid[x, y, z] == PREDATOR:
                    depredadores[0].append(x)
                    depredadores[1].append(y)
                    depredadores[2].append(z)
                elif grid[x, y, z] == OBSTACLE:
                    obstaculos[0].append(x)
                    obstaculos[1].append(y)
                    obstaculos[2].append(z)
    
    # Crear scatter plots
    if peces[0]:
        ax.scatter(peces[0], peces[1], peces[2], 
                   c='cyan', s=20, alpha=0.7, label='Peces', depthshade=True)
    
    if depredadores[0]:
        ax.scatter(depredadores[0], depredadores[1], depredadores[2], 
                   c='red', s=50, alpha=0.9, label='Depredadores', depthshade=True)
    
    if obstaculos[0]:
        ax.scatter(obstaculos[0], obstaculos[1], obstaculos[2], 
                   c='gray', s=40, alpha=0.5, label='Obstáculos', depthshade=True)
    
    # Configuración del gráfico
    ax.set_title(f'Simulación de Cardumen 3D - Paso: {paso}', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, TAMAÑO)
    ax.set_ylim(0, TAMAÑO)
    ax.set_zlim(0, TAMAÑO)
    
    # Leyenda
    ax.legend(loc='upper right')
    
    # Ángulo de visualización
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)  # Mantener la ventana abierta medio segundo por paso
    plt.close()

def visualizar_3d_animado():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_facecolor((0/255, 20/255, 50/255))  # Azul marino oscuro
    #fig.patch.set_facecolor((0/255, 20/255, 50/255))  # Fondo de la figura
    # Cambiar el color de fondo de los paneles 3D (cubos donde nadan los peces) - Matplotlib moderno
    ax.xaxis.set_pane_color((100/255, 150/255, 200/255, 1.0))
    ax.yaxis.set_pane_color((100/255, 150/255, 200/255, 1.0))
    ax.zaxis.set_pane_color((100/255, 150/255, 200/255, 1.0))

    ax.set_title('Simulación de Cardumen 3D', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, TAMAÑO)
    ax.set_ylim(0, TAMAÑO)
    ax.set_zlim(0, TAMAÑO)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()

    peces_scatter = ax.scatter([], [], [], c='cyan', s=20, alpha=0.7, label='Peces', depthshade=True)
    depredadores_scatter = ax.scatter([], [], [], c='red', s=50, alpha=0.9, label='Depredadores', depthshade=True)
    obstaculos_scatter = ax.scatter([], [], [], c='gray', s=40, alpha=0.5, label='Obstáculos', depthshade=True)
    ax.legend(loc='upper right')

    for paso in range(PASOS):
        simular_paso()
        peces = [[], [], []]
        depredadores = [[], [], []]
        obstaculos = [[], [], []]
        for z in range(TAMAÑO):
            for y in range(TAMAÑO):
                for x in range(TAMAÑO):
                    if grid[x, y, z] == FISH:
                        peces[0].append(x)
                        peces[1].append(y)
                        peces[2].append(z)
                    elif grid[x, y, z] == PREDATOR:
                        depredadores[0].append(x)
                        depredadores[1].append(y)
                        depredadores[2].append(z)
                    elif grid[x, y, z] == OBSTACLE:
                        obstaculos[0].append(x)
                        obstaculos[1].append(y)
                        obstaculos[2].append(z)
        peces_scatter._offsets3d = (peces[0], peces[1], peces[2])
        depredadores_scatter._offsets3d = (depredadores[0], depredadores[1], depredadores[2])
        obstaculos_scatter._offsets3d = (obstaculos[0], obstaculos[1], obstaculos[2])
        ax.set_title(f'Simulación de Cardumen 3D - Paso: {paso+1}', fontsize=14)
        plt.pause(0.2)
    plt.show()

if __name__ == "__main__":
    # Parámetros de la simulación
    NUM_FISH = 100
    NUM_PREDATORS = 5
    NUM_OBSTACLES = 20

    # Inicializar simulación
    print("Inicializando simulación 3D de cardumen...")
    inicializar_entidades(NUM_FISH, NUM_PREDATORS, NUM_OBSTACLES)
    print(f"Peces: {len(fish_positions)}, Depredadores: {len(predator_positions)}, Obstáculos: {len(obstacle_positions)}")

    # Bucle principal de simulación
    print("Iniciando simulación...")
    visualizar_3d_animado()
    print("Simulación completada!")