class DivReward:
    def calculate(self, new_fitness, old_fitness):
        return 1.0 - float(new_fitness + 1) / (old_fitness + 1)
