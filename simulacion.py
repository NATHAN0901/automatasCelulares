import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time


TAMAÑO = 25  # Reducido para mejor visualización
PASOS = 30   # Menos pasos para prueba rápida

# Estados celulares
SAN0 = 0     # Tejido sano
TUMOR1 = 1   # Tumor primario
MIGRA2 = 2   # Células migratorias
DEGRA3 = 3   # Matriz degradada
META4 = 4    # Micrometástasis

# Inicializar grid 3D
grid = np.zeros((TAMAÑO, TAMAÑO, TAMAÑO), dtype=np.uint8)

# Colocar tumor primario en el centro
centro = TAMAÑO // 2
grid[centro-1:centro+2, centro-1:centro+2, centro-1:centro+2] = TUMOR1
print(f"Tumor inicial: {np.sum(grid == TUMOR1)} células")


def obtener_vecinos_3d(i, j, k, incluir_diagonales=True):
    """Obtiene vecinos 3D (6 u 26 según configuración)"""
    vecinos = []
    rango = [-1, 0, 1]
    for dx in rango:
        for dy in rango:
            for dz in rango:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Saltar la celda actual
                if not incluir_diagonales and abs(dx) + abs(dy) + abs(dz) > 1:
                    continue  # Solo vecinos directos 
                
                x, y, z = i + dx, j + dy, k + dz
                if 0 <= x < TAMAÑO and 0 <= y < TAMAÑO and 0 <= z < TAMAÑO:
                    vecinos.append((x, y, z))
    
    random.shuffle(vecinos)
    return vecinos

def simular_paso_3d(grid):
    nuevo_grid = grid.copy()
    cambios = {
        'migracion': 0,
        'degradacion': 0,
        'metastasis': 0,
        'crecimiento': 0
    }
    
    # Primera pasada: identificar todas las células activas
    celulas_activas = []
    for i in range(TAMAÑO):
        for j in range(TAMAÑO):
            for k in range(TAMAÑO):
                if grid[i, j, k] in [TUMOR1, MIGRA2, META4]:
                    celulas_activas.append((i, j, k, grid[i, j, k]))
    
    # Procesar solo células activas
    for pos in celulas_activas:
        i, j, k, celda = pos
        
        # 1. Movimiento de células migratorias
        if celda == MIGRA2:
            vecinos = obtener_vecinos_3d(i, j, k, incluir_diagonales=False)
            
            # Intentar moverse
            for x, y, z in vecinos:
                if grid[x, y, z] in [SAN0, DEGRA3]:
                    if random.random() < 0.7:  # Alta probabilidad de movimiento
                        nuevo_grid[x, y, z] = MIGRA2
                        nuevo_grid[i, j, k] = DEGRA3
                        cambios['migracion'] += 1
                        cambios['degradacion'] += 1
                        break
            
            # Intravasación (formación de metástasis)
            if cambios['migracion'] == 0:  # Solo si no se movió
                if i <= 1 or i >= TAMAÑO-2 or j <= 1 or j >= TAMAÑO-2 or k <= 1 or k >= TAMAÑO-2:
                    if random.random() < 0.4:  # Mayor probabilidad
                        # Buscar posición aleatoria lejos de bordes
                        x, y, z = random.randint(3, TAMAÑO-4), random.randint(3, TAMAÑO-4), random.randint(3, TAMAÑO-4)
                        if nuevo_grid[x, y, z] == SAN0:
                            nuevo_grid[x, y, z] = META4
                            cambios['metastasis'] += 1
        
        # 2. Crecimiento tumoral
        elif celda in [TUMOR1, META4]:
            vecinos = obtener_vecinos_3d(i, j, k, incluir_diagonales=False)
            for x, y, z in vecinos:
                if nuevo_grid[x, y, z] == SAN0 and random.random() < 0.3:  # Mayor probabilidad
                    nuevo_grid[x, y, z] = celda
                    cambios['crecimiento'] += 1
        
        # 3. Transición a célula migratoria (EMT)
        elif celda == TUMOR1:
            vecinos = obtener_vecinos_3d(i, j, k)
            vecinos_tumor = sum(1 for x, y, z in vecinos if grid[x, y, z] in [TUMOR1, MIGRA2])
            
            # Condición más relajada para EMT
            if vecinos_tumor < 20 and random.random() < 0.15:  # Mayor probabilidad
                nuevo_grid[i, j, k] = MIGRA2
                cambios['migracion'] += 1
    
    print(f"Cambios: Migración={cambios['migracion']}, Degradación={cambios['degradacion']}, "
          f"Metástasis={cambios['metastasis']}, Crecimiento={cambios['crecimiento']}")
    return nuevo_grid



def visualizar_3d(grid, paso):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Preparar datos
    coords = {TUMOR1: [[], [], []], MIGRA2: [[], [], []], 
              DEGRA3: [[], [], []], META4: [[], [], []]}
    
    # Recopilar coordenadas
    for i in range(TAMAÑO):
        for j in range(TAMAÑO):
            for k in range(TAMAÑO):
                estado = grid[i, j, k]
                if estado in coords:
                    coords[estado][0].append(i)
                    coords[estado][1].append(j)
                    coords[estado][2].append(k)
    
    # Crear scatter plots
    scatter_params = {
        TUMOR1: {'color': 'green', 's': 30, 'alpha': 0.8, 'label': 'Tumor primario'},
        MIGRA2: {'color': 'yellow', 's': 20, 'alpha': 0.9, 'label': 'Células migratorias'},
        DEGRA3: {'color': 'brown', 's': 15, 'alpha': 0.7, 'label': 'Matriz degradada'},
        META4: {'color': 'red', 's': 25, 'alpha': 0.9, 'label': 'Metástasis'}
    }
    
    for estado, params in scatter_params.items():
        if coords[estado][0]:  # Solo si hay puntos
            ax.scatter(
                coords[estado][0], coords[estado][1], coords[estado][2],
                c=params['color'], s=params['s'], alpha=params['alpha'],
                label=params['label'], depthshade=True
            )
    
    # Configuración
    ax.set_title(f'Paso: {paso} - Células Migratorias: {len(coords[MIGRA2][0])}', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, TAMAÑO)
    ax.set_ylim(0, TAMAÑO)
    ax.set_zlim(0, TAMAÑO)
    
    # Leyenda
    ax.legend(loc='upper right')
    
    # Ángulo de visualización puden des 40 y 60
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)  # Importante para liberar memoria ojito




print("Iniciando simulación 3D...")
for paso in range(PASOS):
    start_time = time.time()
    grid = simular_paso_3d(grid)
    elapsed = time.time() - start_time
    
    print(f"Paso {paso+1}/{PASOS} completado en {elapsed:.2f}s - "
          f"Migratorias: {np.sum(grid == MIGRA2)} - "
          f"Metástasis: {np.sum(grid == META4)}")
    
    # Visualizar en cada paso crítico
    if paso in [0, 2, 5] or paso % 10 == 0 or paso == PASOS-1:
        visualizar_3d(grid, paso)

print("Simulación completada!")