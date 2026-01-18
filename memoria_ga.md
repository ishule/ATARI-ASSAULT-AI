**Algoritmos Genéticos — Implementación extensa**

**`src/GeneticAlgorithm/GeneticAlgorithm.cpp`**
- Inicialización / constructores:
  - Existen dos caminos claros: constructor para "evolución de pesos" que recibe una `topology` fija y constructor para "neuroevolución" que recibe `inputSize` y `outputSize` y genera individuos con topologías aleatorias (activa `config.evolveArchitecture`).
  - En ambos casos la población se inicializa con `population.emplace_back(...)`, reservando espacio con `population.reserve(config.populationSize)`.
- Flujo de ejecución principal:
  - `evolve(...)` configura la función de fitness y delega en `evolveWithCustomFitness(...)`.
  - `evolveWithCustomFitness(...)` itera hasta `config.maxGenerations` llamando `runGeneration()` por generación y registrando historiales (`bestFitnessHistory`, `avgFitnessHistory`).
  - Si `bestFit >= config.targetFitness` la evolución termina (early stopping).
- Dentro de cada generación (`runGeneration()`):
  - Evaluación de la población: `evaluatePopulation` (dataset X/Y) o `evaluatePopulationCustom` (función pasada por el usuario).
  - Ordenación por fitness: `sortPopulation()` y guardado de estadística para impresión.
  - Selección de padres: `selection()` soporta `TOURNAMENT`, `ROULETTE`, `RANK` y preserva élite (`eliteRatio`).
  - Reproducción: `breed(parents)` crea la nueva población; mantiene élite, selecciona pares de padres aleatoriamente y aplica `crossover` (pesos) o `crossoverNeuroevolution` (estructura + pesos) según `config.evolveArchitecture`.
  - Guardado y reporting: `saveBest(...)` escribe el mejor individuo al finalizar; `printStats()` muestra `Gen`, `Best`, `Avg`, además de `Arch` y `Params` si se evoluciona arquitectura.
  - Notas de diseño:
  - `selection()` construye una lista de padres que luego se usa para cruces; el código usa `tournamentSelection`, `rouletteSelection` o `rankSelection` con implementaciones estándar y pequeñas defensas (p. ej. offset en ruleta para fitness negativos).
  - `breed()` gestiona diferencias entre modos usando `config.evolveArchitecture` y crea hijos hasta rellenar la población.

**`src/GeneticAlgorithm/Individual.cpp`**
- Representación y estructura:
  - `Individual` contiene `topology_` (vector<int>), `weights_` (VecWeights: [capa][neurona][peso]), `biases_`, `activation_`, `fitness_`.
  - Las dimensiones se mantienen de forma explícita: para `i` en `topology_`, la conexión `i -> i+1` produce una matriz de tamaño `topology_[i+1] x topology_[i]`.
- Inicialización:
  - `initializeRandomWeights()` implementa inicialización tipo Xavier/Glorot: límite = sqrt(6/(in+out)) y distribución uniforme en [-limit, limit].
  - Para neuroevolución existe `generateRandomTopology(...)` que construye una topología aleatoria entre `minHiddenLayers`/`maxHiddenLayers` y `minNeurons`/`maxNeurons`.
- Cruce y mezcla de soluciones:
  - `crossover(const Individual& other, double mutationRate)`: exige misma arquitectura; mezcla pesos y biases por elemento, con probabilidad `mutationRate` para mutar (ruido normal) o heredar aleatoriamente del padre A o B.
  - `crossoverNeuroevolution(const Individual& other, double mutationRate, double archMutationRate)`: diseñado para topologías distintas.
    - Selecciona el "mejor" padre como base (por fitness) y el "peor" como complemento.
    - Con probabilidad `archMutationRate` puede modificar la arquitectura (varias estrategias: quitar/insertar capa, añadir/quitar neuronas, ajustar neuronas por capa).
    - Para cada entrada de peso intenta heredar de los padres cuando existe; si no existe (debido a diferente topología) inicializa por Xavier.
    - Mezcla pesos prefiriendo el mejor padre, con mutaciones y pequeñas heredabilidades del peor padre para diversidad.
- Mutaciones:
  - `mutateWeights(double mutationRate, double mutationStrength)`: aplica perturbación Gaussiana por peso con clipping a rangos razonables (p. ej. [-5,5] para pesos, [-2,2] para biases).
  - `mutateArchitecture(...)`: operaciones estructurales (añadir/eliminar neuronas, insertar/eliminar capas). El código actual cubre casos de inserción y eliminación y adapta las matrices de pesos/biases (reconstruyendo o redimensionando y re-inicializando conexiones cuando es necesario).
- Forward / evaluación:
  - `forwardPass(...)` implementa el pase hacia delante iterando capas; aplica `ActivationFunctions::apply` en capas ocultas y `sigmoide` en la salida (según implementación actual).

**Diferencias prácticas y consecuencias: Neuroevolución vs Evolución solo de pesos**
- Espacio de búsqueda:
  - Evolución solo de pesos: el espacio está limitado a los pesos/biases de una arquitectura fija. Es un espacio continuo de alta dimensión pero con topología conocida.
  - Neuroevolución: el espacio incluye combinaciones discretas de topologías (número de capas, neuronas por capa) y los pesos; es muy superior en tamaño y mixto (discreto + continuo).
- Operadores genéticos:
  - Pesos-only: usa `crossover` y `mutateWeights`. El crossover requiere que ambos padres compartan la misma `topology_`.
  - Neuroevolución: usa `crossoverNeuroevolution` que puede mezclar arquitecturas, y `mutateArchitecture` para cambiar la estructura. Cuando una conexión no existe en los padres, el código inicializa pesos por Xavier.
- Convergencia y diversidad:
  - Neuroevolución introduce mayor diversidad estructural, facilita escapar de óptimos locales relacionados con arquitectura y permite encontrar topologías más adecuadas, pero tiene mayor coste computacional por individuo (evaluación más cara) y requiere control fino de `archMutationRate`.
  - Evolución de pesos converge más rápido cuando la arquitectura ya es adecuada, y el ajuste fino suele ser más estable con tasas de mutación pequeñas.
- Implementación en este repositorio (consecuencias directas):
  - El `GeneticAlgorithm` alterna entre ambos modos por la bandera `config.evolveArchitecture` y adapta el flujo de `breed` escogiendo el operador correspondiente.
  - El código de `crossoverNeuroevolution` está escrito para preservar conocimiento del "mejor" padre y rellenar con inicializaciones Xavier cuando aparece nueva conectividad: esto reduce el error de incompatibilidad estructural, pero puede introducir parámetros aleatorios que necesitan más generaciones para optimizar.
  - Para selección de padres, `rouletteSelection()` ya contiene un offset para manejar fitness negativos; en neuroevolución (donde fitness puede variar ampliamente) es recomendable usar `TOURNAMENT` o `RANK` para estabilidad.
- Recomendaciones prácticas de configuración:
  - Para búsqueda de topologías iniciales: usar poblaciones moderadas (50–200), `archMutationRate` bajo-moderado (0.01–0.1) y `mutationRate` para pesos algo mayor mientras se exploran topologías.
  - Para afinado de pesos en arquitectura conocida: usar `evolveArchitecture=false`, disminuir `mutationRate` gradualmente, y aumentar `eliteRatio` para conservar mejores soluciones.
  - Usar `tournamentSelection` con `tournamentSize=3..7` para equilibrar presión selectiva sin perder diversidad.

**Conclusión breve**
La implementación del repositorio separa claramente los dos modos: una ruta optimizada y más rápida cuando la arquitectura es fija (evolución de pesos), y una ruta más general y exploratoria que altera la arquitectura (neuroevolución). Los puntos críticos a vigilar son: la inicialización de conexiones cuando cambia la topología (Xavier implementado), la correcta adaptación de las matrices de pesos/biases al mutar o insertar capas, y la configuración de tasas de mutación/selección que controlan la exploración vs explotación.
