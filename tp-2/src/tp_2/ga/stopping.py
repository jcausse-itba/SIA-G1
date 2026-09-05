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

        criterion = self.config.get("stop_criterion", "generations")

        # 1. Cantidad máxima de generaciones (criterio directo o límite de seguridad)
        max_gen = self.config.get("max_generations", 1000)
        if generation >= max_gen:
            return True, f"Alcanzado el límite máximo de generaciones ({max_gen})"

        # 2. Fitness objetivo
        if criterion == "fitness":
            target_fit = self.config.get("target_fitness", None)
            if target_fit is not None and best_fitness >= target_fit:
                return True, f"Alcanzado el fitness objetivo ({target_fit})"

        # 3. Límite de tiempo
        if criterion == "time":
            max_time = self.config.get("max_time_seconds", None)
            if max_time is not None and (time.time() - self.start_time) >= max_time:
                return True, f"Alcanzado el tiempo máximo ({max_time}s)"

        # 4. Estancamiento (Estructura/Contenido)
        if criterion in ("structure", "content"):
            stag_limit = self.config.get("stagnation_limit", None)
            if stag_limit is not None and len(self.best_fitness_history) > stag_limit:
                recent = self.best_fitness_history[-stag_limit:]
                if max(recent) - min(recent) < 1e-5:
                    return True, f"Estancamiento detectado en las últimas {stag_limit} generaciones"

        return False, ""