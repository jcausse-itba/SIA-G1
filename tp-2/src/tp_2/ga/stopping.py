import time
from typing import Any, Tuple

class StoppingCriteria:
    """Verificador de condiciones de terminación para la evolución."""

    def __init__(self, config: Any):
        self.config = config
        self.start_time = time.time()
        self.best_fitness_history = []

    def should_stop(self, generation: int, best_fitness: float) -> Tuple[bool, str]:
        self.best_fitness_history.append(best_fitness)

        # 1. Cantidad máxima de generaciones
        max_gen = getattr(self.config, 'max_generations', 100000000)
        if generation >= max_gen:
            return True, f"Alcanzado el límite máximo de generaciones ({max_gen})"

        # 2. Fitness objetivo
        target_fit = getattr(self.config, 'target_fitness', None)
        if target_fit is not None and best_fitness >= target_fit:
            return True, f"Alcanzado el fitness objetivo ({target_fit})"

        # 3. Límite de tiempo
        max_time = getattr(self.config, 'max_time_seconds', None)
        if max_time is not None and (time.time() - self.start_time) >= max_time:
            return True, f"Alcanzado el tiempo máximo ({max_time}s)"

        # 4. Estancamiento (Estructura/Contenido)
        stag_limit = getattr(self.config, 'stagnation_limit', None)
        if stag_limit is not None and len(self.best_fitness_history) > stag_limit:
            recent = self.best_fitness_history[-stag_limit:]
            if max(recent) - min(recent) < 1e-5:
                return True, f"Estancamiento detectado en las últimas {stag_limit} generaciones"

        return False, ""