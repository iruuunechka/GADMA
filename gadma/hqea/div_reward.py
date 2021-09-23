class DivReward:
    def calculate(self, new_fitness, old_fitness, coef):
        return (1.0 - float(new_fitness + 1) / (old_fitness + 1)) / (coef ** 6)
