"""Registro de todos los algoritmos: descripción, propiedades teóricas y parámetros."""
from __future__ import annotations

from baselines.aco import run_aco
from common.problem import AlgorithmSpec, ParamSpec as P
from baselines.drl import run_drl
from baselines.grid_search import run_astar, run_dijkstra, run_theta
from baselines.hybrid_astar import run_hybrid_astar
from baselines.potential import run_potential
from baselines.sampling import run_prm, run_rrt, run_rrt_star
from baselines.slam import run_slam
from baselines.wdt.adapter import run_tessellation
from baselines.topological import run_topological
from tve.adapter import run_tve
from baselines.visibility import run_visibility

CELL = P("celda", "Resolución de malla", 2, "int", 1, 12, 1, suffix=" px",
         tip="Tamaño de cada celda de la malla en píxeles. 1 = máxima precisión (más lento); "
             "valores mayores reducen el costo pero la ruta se aleja de los obstáculos finos.")
SEED = P("semilla", "Semilla", 0, "int", 0, 99999, 1, tip="Fija la aleatoriedad para poder repetir el resultado.")

PROPS_KEYS = [("completitud", "Completitud"), ("optimalidad", "Optimalidad"), ("determinista", "Determinista"),
              ("mapa", "Requiere mapa previo"), ("cinematica", "Cinemática"), ("tipo", "Tipo")]

SPECS: list[AlgorithmSpec] = [
    AlgorithmSpec(
        "teselado", "Teselado (artículo)", "🧩", "Método del artículo",
        "Teselado por frentes de onda, esqueleto ponderado, reducción y Bézier.",
        "Método de Ladino et al.: dilata los obstáculos, expande frentes de onda cuadrados desde cada "
        "obstáculo, toma las fronteras entre teselas (esqueleto de Voronoi) y sus nodos, genera la ruta voraz "
        "y n−1 alternativas, la reduce con rectas y la suaviza con curvas de Bézier.",
        {"completitud": "Completo (con puentes de respaldo)", "optimalidad": "No (aproximada)",
         "determinista": "Sí", "mapa": "Sí", "cinematica": "Holonómica, curva suave G1", "tipo": "Global, grafo"},
        [P("generadores", "Generadores", "obstacles", "choice", choices=("obstacles", "segments"),
           tip="obstacles = artículo; segments = divide paredes (interiores)."),
         P("suavizado", "Suavizado", "piecewise", "choice", choices=("piecewise", "global", "none"),
           tip="Bézier cúbicas C1 (piecewise), Bézier global de grado n, o sin suavizar.")],
        lambda pr, **k: run_tessellation(pr, optimizar=False, **k), True),
    AlgorithmSpec(
        "teselado_opt", "Teselado optimizado", "🧩", "Método del artículo",
        "El método del artículo con las optimizaciones propias (elección tras reducir + tensado).",
        "Igual que el método del artículo, pero elige la ruta más corta DESPUÉS de reducir (incluye Dijkstra "
        "sobre el esqueleto como candidata) y aplica un tensado final de vértices.",
        {"completitud": "Completo (con puentes de respaldo)", "optimalidad": "Casi óptima (localmente)",
         "determinista": "Sí", "mapa": "Sí", "cinematica": "Holonómica, curva suave G1", "tipo": "Global, grafo"},
        [P("generadores", "Generadores", "obstacles", "choice", choices=("obstacles", "segments"),
           tip="obstacles = artículo; segments = divide paredes (interiores)."),
         P("suavizado", "Suavizado", "piecewise", "choice", choices=("piecewise", "global", "none"),
           tip="Bézier cúbicas C1 (piecewise), Bézier global de grado n, o sin suavizar.")],
        lambda pr, **k: run_tessellation(pr, optimizar=True, **k), True),
    AlgorithmSpec(
        "tve", "TVE (mejorado)", "⚡", "Método del artículo",
        "Versión mejorada del artículo: Voronoi en una pasada, caché, Dijkstra + K rutas, embudo y arcos.",
        "<b>TVE — Teselado Voronoi + Embudo</b>. Conserva la idea del artículo (teselar desde los obstáculos, "
        "navegar por el esqueleto y reducir con las líneas transversales) pero cambia cada etapa:<br>"
        "1) el teselado es un Voronoi euclidiano calculado en UNA pasada (transformada de distancia con "
        "etiquetas) en vez de expandir frentes iterativamente;<br>"
        "2) teselado y esqueleto se guardan en caché: nuevas consultas en el mismo mapa sólo conectan S y T;<br>"
        "3) rutas: Dijkstra sobre el esqueleto + K alternativas por penalización (rodean obstáculos distintos), "
        "en vez de la ruta voraz + n−1;<br>"
        "4) reducción: cada punto del esqueleto equidista de dos obstáculos; el segmento entre ambos es un "
        "<i>portal</i> libre (la línea transversal del artículo) y el <b>algoritmo del embudo</b> da la ruta más "
        "corta que los cruza en O(n), sin ciclos de reducción ni el parámetro D (también con portales "
        "«disco»: diámetros del disco libre de cada punto del esqueleto);<br>"
        "S y T se conectan con varios conectores (uno por sector) para que la ruta no retroceda;<br>"
        "5) colisiones: aceptación rápida con la transformada de distancia exacta (bolas libres) y prueba "
        "exacta sólo cerca de obstáculos; poda de candidatas con una cota inferior (sin ejecutar el embudo);<br>"
        "6) suavizado con <b>arcos de radio R_min</b> (curvatura acotada) para vehículos tipo auto;<br>"
        "7) generadores automáticos: si el esqueleto no cubre una región (p.ej. laberinto de un solo "
        "obstáculo) se dividen los contornos, evitando puentes costosos en cada consulta.",
        {"completitud": "Completo (con puentes de respaldo)", "optimalidad": "Casi óptima (geodésica en el corredor)",
         "determinista": "Sí", "mapa": "Sí", "cinematica": "Holonómica y Ackermann (arcos de radio R)",
         "tipo": "Global, grafo + embudo"},
        [P("candidatas", "Rutas candidatas K", 4, "int", 1, 12, 1,
           tip="Rutas alternativas del esqueleto que se evalúan con el embudo; se queda la más corta."),
         P("holgura_extra_m", "Holgura extra", 0.10, "float", 0.0, 2.0, 0.02, suffix=" m",
           tip="Margen de los portales: la ruta se mantiene al menos esta distancia de los obstáculos dilatados."),
         P("suavizado", "Suavizado", "arcos", "choice", choices=("arcos", "bezier", "ninguno"),
           tip="arcos = filetes circulares de radio R (curvatura acotada, Ackermann); bezier = cúbicas C1."),
         P("radio_suavizado_m", "Radio de los arcos", 1.0, "float", 0.05, 20, 0.05, suffix=" m",
           tip="Radio de giro deseado; si no cabe, el arco se reduce lo necesario."),
         P("portales", "Portales del embudo", "ambos", "choice", choices=("ambos", "teselas", "disco"),
           tip="teselas = cuerda entre los obstáculos más cercanos de las dos teselas (rutas más ceñidas); "
               "disco = diámetro del disco libre de cada punto del esqueleto (más robusto); "
               "ambos = evalúa los dos y se queda con la ruta más corta."),
         P("generadores", "Generadores", "auto", "choice", choices=("auto", "obstacles", "segments"),
           tip="auto = por obstáculo y, si el esqueleto queda incompleto, divide contornos; "
               "obstacles = artículo; segments = siempre divide paredes."),
         P("usar_cache", "Usar caché multi-consulta", True, "bool",
           tip="Reutiliza teselado y esqueleto si el mapa y el robot no cambiaron.")],
        run_tve, True),
    AlgorithmSpec(
        "dijkstra", "Dijkstra", "🔷", "Búsqueda en grafos",
        "Expande los nodos en orden de costo acumulado: óptimo en la malla, sin heurística.",
        "Dijkstra (1959) explora la malla como una onda que crece desde S en todas las direcciones, en orden "
        "de costo acumulado g(n). Es completo y encuentra la ruta más corta sobre la malla 8-conexa, pero "
        "expande muchos nodos porque no sabe dónde está T. La ruta sigue los 8 rumbos de la malla.",
        {"completitud": "Completo (resolución)", "optimalidad": "Óptimo en la malla", "determinista": "Sí",
         "mapa": "Sí", "cinematica": "Holonómica (esquinas de 45°)", "tipo": "Global, malla"},
        [CELL, P("diagonal", "Movimientos diagonales", True, "bool", tip="8-conexa (sí) o 4-conexa (no).")],
        run_dijkstra, True),
    AlgorithmSpec(
        "astar", "A*", "⭐", "Búsqueda en grafos",
        "Dijkstra guiado por una heurística admisible: misma ruta óptima con muchas menos expansiones.",
        "A* (Hart, Nilsson y Raphael, 1968) ordena los nodos por f(n) = g(n) + w·h(n), donde h es la distancia "
        "octil a T (admisible y consistente). Con w = 1 garantiza la ruta óptima de la malla expandiendo mucho "
        "menos que Dijkstra. Con w > 1 (A* ponderado) es aún más rápido, pero la ruta puede ser hasta w veces "
        "más larga.",
        {"completitud": "Completo (resolución)", "optimalidad": "Óptimo en la malla (w = 1)", "determinista": "Sí",
         "mapa": "Sí", "cinematica": "Holonómica (esquinas de 45°)", "tipo": "Global, malla"},
        [CELL, P("peso", "Peso de la heurística w", 1.0, "float", 1.0, 5.0, 0.1,
                 tip="1 = óptimo. Mayor = más rápido pero sub-óptimo (A* ponderado)."),
         P("diagonal", "Movimientos diagonales", True, "bool", tip="8-conexa (sí) o 4-conexa (no).")],
        run_astar, True),
    AlgorithmSpec(
        "theta", "Theta*", "📐", "Búsqueda en grafos",
        "A* de ángulo libre: conecta con el abuelo si hay línea de vista; rutas casi óptimas.",
        "Theta* (Nash, Daniel, Koenig y Felner, 2007) es una variante de A*. Al expandir un nodo, intenta "
        "conectar al vecino directamente con el padre de su padre cuando hay línea de vista. Así la ruta deja "
        "de estar atada a los rumbos de la malla y resulta mucho más corta y con menos giros, al costo de "
        "verificar la línea de vista.",
        {"completitud": "Completo (resolución)", "optimalidad": "Casi óptima (cualquier ángulo)",
         "determinista": "Sí", "mapa": "Sí", "cinematica": "Holonómica (poligonal)", "tipo": "Global, malla"},
        [CELL], run_theta, True),
    AlgorithmSpec(
        "rrt", "RRT", "🌳", "Muestreo",
        "Árbol aleatorio que crece rápidamente hacia regiones inexploradas.",
        "RRT (LaValle, 1998) hace crecer un árbol desde S hacia puntos aleatorios del espacio libre (con cierta "
        "probabilidad hacia T). Cada nuevo nodo avanza un paso desde el nodo más cercano si el tramo no "
        "choca. Es probabilísticamente completo y muy rápido en espacios amplios o de muchas dimensiones, "
        "pero sus rutas son quebradas y lejos del óptimo.",
        {"completitud": "Probabilísticamente completo", "optimalidad": "No", "determinista": "No (aleatorio)",
         "mapa": "Sí", "cinematica": "Holonómica (poligonal)", "tipo": "Global, muestreo"},
        [P("paso_m", "Paso", 0.5, "float", 0.05, 5.0, 0.05, suffix=" m", tip="Longitud máxima de cada rama nueva."),
         P("iteraciones", "Iteraciones máx.", 6000, "int", 100, 100000, 100, tip="Muestras antes de rendirse."),
         P("sesgo_meta", "Sesgo hacia T", 0.08, "float", 0.0, 1.0, 0.01,
           tip="Probabilidad de muestrear directamente T (acelera la llegada)."),
         SEED], run_rrt, True),
    AlgorithmSpec(
        "rrt_star", "RRT*", "🌲", "Muestreo",
        "RRT con elección del mejor padre y recableado: asintóticamente óptimo.",
        "RRT* (Karaman y Frazzoli, 2011) elige para cada nodo nuevo el padre que minimiza el costo desde S, y "
        "«recablea» los vecinos si pasar por el nuevo nodo les conviene. Con suficientes iteraciones la ruta "
        "converge a la óptima (asintóticamente óptimo). Es más lento que RRT.",
        {"completitud": "Probabilísticamente completo", "optimalidad": "Asintóticamente óptimo",
         "determinista": "No (aleatorio)", "mapa": "Sí", "cinematica": "Holonómica (poligonal)",
         "tipo": "Global, muestreo"},
        [P("paso_m", "Paso", 0.5, "float", 0.05, 5.0, 0.05, suffix=" m", tip="Longitud máxima de cada rama nueva."),
         P("iteraciones", "Iteraciones", 4000, "int", 100, 100000, 100,
           tip="Más iteraciones = ruta más corta (sigue mejorando tras hallar la primera)."),
         P("sesgo_meta", "Sesgo hacia T", 0.05, "float", 0.0, 1.0, 0.01, tip="Probabilidad de muestrear T."),
         P("gamma", "Constante del radio γ", 2.5, "float", 0.5, 10, 0.1,
           tip="Escala el radio de vecindad r = γ·(log n / n)^½ para elegir padre y recablear."),
         SEED], run_rrt_star, False),
    AlgorithmSpec(
        "prm", "PRM", "🕸️", "Muestreo",
        "Hoja de ruta probabilística: muestras + conexiones visibles + Dijkstra.",
        "PRM (Kavraki, Švestka, Latombe y Overmars, 1996) toma muestras aleatorias del espacio libre y conecta "
        "cada una con sus k vecinos más cercanos si el tramo no choca. Resulta un grafo (hoja de ruta) que "
        "sirve para muchas consultas S→T; cada consulta se resuelve con Dijkstra.",
        {"completitud": "Probabilísticamente completo", "optimalidad": "No (asint. óptimo en PRM*)",
         "determinista": "No (aleatorio)", "mapa": "Sí", "cinematica": "Holonómica (poligonal)",
         "tipo": "Global, muestreo multi-consulta"},
        [P("muestras", "Muestras", 600, "int", 20, 20000, 50, tip="Nodos aleatorios de la hoja de ruta."),
         P("vecinos", "Vecinos k", 12, "int", 2, 60, 1, tip="Conexiones que se intentan por nodo."), SEED],
        run_prm, True),
    AlgorithmSpec(
        "potential", "Campos potenciales", "🧲", "Reactivos",
        "El robot baja por un potencial: atracción a T + repulsión de obstáculos.",
        "Campos de potencial artificial (Khatib, 1986): la meta atrae y los obstáculos repelen, y el robot sigue "
        "el gradiente descendente. Es barato, suave y apto para control reactivo en línea, pero no es completo: "
        "puede quedar atrapado en mínimos locales (obstáculos cóncavos, pasos estrechos). Aquí escapa con "
        "caminatas aleatorias.",
        {"completitud": "No (mínimos locales)", "optimalidad": "No", "determinista": "Casi (escape aleatorio)",
         "mapa": "Local", "cinematica": "Holonómica, trayectoria suave", "tipo": "Reactivo"},
        [P("rango_repulsion_m", "Rango de repulsión", 0.6, "float", 0.05, 5, 0.05, suffix=" m",
           tip="Distancia a partir de la cual los obstáculos empujan al robot."),
         P("ganancia_repulsion", "Ganancia de repulsión", 1.0, "float", 0.05, 20, 0.05,
           tip="Intensidad de la repulsión frente a la atracción."),
         P("escape_aleatorio", "Escapar de mínimos locales", True, "bool",
           tip="Si se atasca, da pasos aleatorios crecientes para salir."), SEED],
        run_potential, True),
    AlgorithmSpec(
        "visibility", "Grafo de visibilidad", "👁️", "Búsqueda en grafos",
        "Une los vértices convexos visibles entre sí: la ruta euclidiana más corta (referencia).",
        "El grafo de visibilidad (Lozano-Pérez y Wesley, 1979) une con rectas todos los pares de vértices de "
        "obstáculos que se ven entre sí. La ruta más corta en el plano pasa por esos vértices, así que Dijkstra "
        "sobre este grafo da la ruta ÓPTIMA (salvo la aproximación poligonal). Es la referencia de longitud "
        "de la comparación. Sus rutas rozan los obstáculos.",
        {"completitud": "Completo", "optimalidad": "Óptimo (euclidiano)", "determinista": "Sí", "mapa": "Sí",
         "cinematica": "Holonómica (poligonal)", "tipo": "Global, grafo exacto"},
        [P("tolerancia_poligono", "Tolerancia poligonal", 1.0, "float", 0.3, 5, 0.1, suffix=" px",
           tip="Error máximo al aproximar los contornos con polígonos (menos = más vértices).")],
        run_visibility, True),
    AlgorithmSpec(
        "hybrid_astar", "Hybrid A*", "🚗", "No holonómicos",
        "A* en (x, y, θ) con arcos de curvatura acotada: rutas factibles para un auto.",
        "Hybrid A* (Dolgov, Thrun, Montemerlo y Diebel, 2008) busca en el espacio de estados (x, y, θ) de un "
        "vehículo tipo Ackermann aplicando arcos de curvatura |κ| ≤ 1/R_min. Las rutas respetan el radio de "
        "giro mínimo por construcción, así que un auto las puede seguir sin maniobras. Es la opción para "
        "robots NO holonómicos.",
        {"completitud": "Completo (resolución, aprox.)", "optimalidad": "No (casi óptima)", "determinista": "Sí",
         "mapa": "Sí", "cinematica": "Ackermann (curvatura acotada)", "tipo": "Global, espacio de estados"},
        [P("radio_giro_m", "Radio de giro mínimo", 1.0, "float", 0.1, 20, 0.05, suffix=" m",
           tip="Radio de giro mínimo del vehículo."),
         P("paso_m", "Longitud de primitiva", 0.35, "float", 0.05, 5, 0.05, suffix=" m",
           tip="Longitud de cada arco de movimiento."),
         P("curvaturas", "Nº de curvaturas", 5, "int", 3, 11, 2, tip="Arcos entre −1/R y +1/R por expansión."),
         P("reversa", "Permitir reversa", False, "bool", tip="Agrega primitivas marcha atrás (con penalización)."),
         P("orientacion_inicial", "Orientación inicial", "hacia T", "choice",
           choices=("hacia T", "este (0°)", "sur (90°)", "oeste (180°)", "norte (270°)"),
           tip="Hacia dónde mira el vehículo en S.")],
        run_hybrid_astar, False),
    AlgorithmSpec(
        "aco", "Colonia de hormigas", "🐜", "Metaheurísticas",
        "Hormigas virtuales depositan feromona; la colonia converge a rutas cortas.",
        "La optimización por colonia de hormigas (Dorigo, 1992) usa m hormigas que recorren la malla eligiendo "
        "vecinos con probabilidad ∝ τ^α·η^β (feromona × avance hacia T). Las que llegan depositan feromona "
        "inversa a su longitud; la feromona se evapora y la mejor ruta recibe refuerzo elitista. Incluye "
        "retroceso en callejones y poda de rodeos. Es una metaheurística: sin garantías y dependiente de la "
        "semilla.",
        {"completitud": "No garantizada", "optimalidad": "No (metaheurística)", "determinista": "No (aleatorio)",
         "mapa": "Sí", "cinematica": "Holonómica (esquinas de 45°)", "tipo": "Global, metaheurística"},
        [P("celda", "Resolución de malla", 4, "int", 1, 12, 1, suffix=" px", tip="Tamaño de celda de la malla."),
         P("hormigas", "Hormigas", 40, "int", 2, 500, 5, tip="Hormigas por iteración."),
         P("iteraciones", "Iteraciones", 40, "int", 1, 500, 5, tip="Generaciones de la colonia."),
         P("alfa", "α (feromona)", 1.0, "float", 0.0, 5, 0.1, tip="Peso de la feromona."),
         P("beta", "β (heurística)", 2.0, "float", 0.0, 10, 0.1, tip="Peso del avance hacia T."),
         P("evaporacion", "Evaporación ρ", 0.25, "float", 0.01, 0.99, 0.01, tip="Fracción que se evapora."),
         P("elitismo", "Refuerzo elitista", 2.0, "float", 0.0, 20, 0.5, tip="Feromona extra a la mejor ruta."),
         SEED], run_aco, False),
    AlgorithmSpec(
        "topological", "Topológico (homotopía)", "🍩", "Topología",
        "Obstáculos ≅ discos → puntos; clases de homotopía/homología; geodésica de cada clase.",
        "Cada obstáculo es una variedad con borde. Si su característica de Euler χ = 1 es homeomorfo al disco "
        "D² (frontera S¹) y se contrae a un punto. El espacio libre equivale a un disco con n puntos removidos: "
        "π₁ = grupo libre Fₙ (clases de HOMOTOPÍA) y H₁ = ℤⁿ (clases de HOMOLOGÍA). La palabra reducida de "
        "cruces con rayos que salen de cada punto clasifica la homotopía, y su abelianización la homología. "
        "A* en el espacio (celda, palabra) encuentra las K clases más cortas; cada una se tensa sin salir de "
        "su clase y se elige la geodésica más corta.",
        {"completitud": "Completo (resolución)", "optimalidad": "Óptima entre las K clases (tras tensado)",
         "determinista": "Sí", "mapa": "Sí", "cinematica": "Holonómica (poligonal)",
         "tipo": "Global, topológico"},
        [P("celda", "Resolución de malla", 3, "int", 1, 12, 1, suffix=" px", tip="Tamaño de celda de la búsqueda."),
         P("clases", "Clases a encontrar (K)", 6, "int", 1, 30, 1,
           tip="Cuántas clases de homotopía distintas buscar, en orden de longitud."),
         P("longitud_palabra", "Longitud máx. de palabra", 4, "int", 1, 12, 1,
           tip="Límite de cruces reducidos (evita clases que dan vueltas).")],
        run_topological, False),
    AlgorithmSpec(
        "slam", "SLAM", "🛰️", "Aprendizaje y SLAM",
        "Sin mapa previo: LiDAR + odometría con deriva, scan matching, mapa log-odds y A*.",
        "SLAM (localización y mapeo simultáneos) resuelve un problema distinto: el robot NO conoce el mapa. En "
        "cada paso predice su pose con la odometría, la corrige alineando el escaneo LiDAR con el mapa "
        "(scan matching con campo de verosimilitud), actualiza un mapa de ocupación log-odds y replanifica "
        "con A* sobre lo descubierto (lo desconocido se asume libre). La ruta es la trayectoria real "
        "recorrida.",
        {"completitud": "No garantizada (depende de la localización)", "optimalidad": "No (explora)",
         "determinista": "No (ruido de sensores)", "mapa": "No (lo construye)",
         "cinematica": "Holonómica / diferencial", "tipo": "En línea, entorno desconocido"},
        [P("rango_lidar_m", "Alcance del LiDAR", 3.0, "float", 0.5, 30, 0.5, suffix=" m", tip="Alcance máximo."),
         P("haces", "Haces del LiDAR", 180, "int", 16, 720, 4, tip="Resolución angular del escaneo."),
         P("ruido_odometria", "Ruido de avance", 0.02, "float", 0.0, 0.3, 0.01, tip="Error relativo por paso."),
         P("deriva_giro_deg", "Deriva de rumbo", 0.08, "float", 0.0, 2.0, 0.01, suffix=" °/paso",
           tip="Error sistemático de rumbo que se acumula (giroscopio/encoders)."),
         P("localizacion", "Scan matching (L de SLAM)", True, "bool",
           tip="Desactívelo para ver cómo deriva la odometría sin corrección."),
         P("paso_m", "Paso del robot", 0.15, "float", 0.02, 1.0, 0.01, suffix=" m", tip="Avance por iteración."),
         SEED], run_slam, False),
    AlgorithmSpec(
        "drl", "DRL (Double DQN)", "🧠", "Aprendizaje y SLAM",
        "Política neuronal entrenada por refuerzo con observaciones locales; generaliza a mapas nuevos.",
        "Aprendizaje por refuerzo profundo: una red dueling Double DQN elige uno de 8 movimientos a partir de "
        "lo que el robot percibe a su alrededor (ocupación local, memoria de celdas visitadas, vista gruesa, "
        "rayos y dirección a T). Se entrenó una vez en cientos de mapas aleatorios; no planifica, reacciona. "
        "Puede reajustarse al entorno actual con «Entrenar».",
        {"completitud": "No garantizada", "optimalidad": "No", "determinista": "Casi (escape aleatorio)",
         "mapa": "No (sólo percepción local)", "cinematica": "Holonómica (esquinas de 45°)",
         "tipo": "Reactivo aprendido"},
        [SEED, P("max_pasos", "Pasos máx. (0 = auto)", 0, "int", 0, 20000, 50, tip="Límite de pasos del episodio.")],
        run_drl, True),
]

ORDER = ["teselado", "teselado_opt", "tve", "dijkstra", "astar", "theta", "visibility", "rrt", "rrt_star", "prm",
         "potential", "hybrid_astar", "aco", "topological", "slam", "drl"]
SPECS.sort(key=lambda s: ORDER.index(s.key))
BY_KEY = {s.key: s for s in SPECS}
FAMILIES = []
for s in SPECS:
    if s.family not in FAMILIES:
        FAMILIES.append(s.family)
