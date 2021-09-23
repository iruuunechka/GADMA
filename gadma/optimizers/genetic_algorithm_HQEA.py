import numpy as np

from . import GeneticAlgorithm, Optimizer
from .global_optimizer import register_global_optimizer
from ..hqea import GreedyQAgent, DivReward, BetterCountState
from ..hqea.better_count_paired_state import BetterCountPairedState
from ..utils import WeightedMetaArray
from ..utils import sort_by_other_list


class GeneticAlgorithmHQEA(GeneticAlgorithm):
    """
    Class for Genetic Algorithm combined with HQEA.

    :param gen_size: Size of generation of genetic algorithm. That is number
                     of individuals/solutions on each step of GA.
    :type gen_size: int
    :param n_elitism: Number of best models from previous generation in GA
                      that will be taken to new iteration.
    :type n_elitism: int
    :param p_mutation: probability of mutation in one generation of GA.
    :type p_mutation: float
    :param p_crossover: probability of crossover in one generation of GA.
    :type p_crossover: float
    :param p_random: Probability of random generated individual in one
                     generation of GA.
    :type p_random: float
    :param mut_rate: Initial mean mutation rate.
    :type mut_rate: float
    :param mut_strength: initial mutation "strength" - mean fraction of
                         model parameters that will be mutated.
    :type mut_strength: float
    :param const_mut_rate: constant to change mutation rate according to
                           one-fifth algorithm. Check GADMA paper for more
                           information.
    :type const_mut_rate: float
    :param eps: const for model's log likelihood compare.
                Model is better if its log likelihood is greater than
                log likelihood of another model by epsilon.
    :type eps: float
    :param n_stuck_gen: Number of iterations for GA stopping: GA stops when
                        it can't improve model during n_stuck_gen generations.
    :type n_stuck_gen: int
    :param selection_type: Type of selection operator in GA. Could be:
                           * 'roulette_wheel'
                           * 'rank'
                           See help(GeneticAlgorithm.selection) for more
                           information.
    :type selection_type: str
    :param selection_random: If True then number of mutants and crossover's
                             offsprings in new generation will be binomial
                             random variable.
    :type selection_type: bool
    :param mutation_type: Type of mutation operator in GA. Could be:
                          * 'uniform'
                          * 'resample'
                          * 'gaussian'
                          See help(GeneticAlgorithm.mutation) for more
                          information.
    :type mutation_type: str
    :param one_fifth_rule: If True then one fifth rule is used in mutation.
    :type one_fifth_rule: bool
    :param crossover_type: Type of crossover operator in GA. Could be:
                           * 'k-point'
                           * 'uniform'
                           See help(GeneticAlgorithm.crossover) for more
                           information.
    :type crossover_type: str
    :param crossover_k: k for 'k-point' crossover type.
    :type crossover_k: int
    :param random_type: Type of random generation of new offsprings. Could be:
                        * 'uniform'
                        * 'resample'
                        * 'custom'
                        See help(GlobalOptimizer.randomize) for more
                        information.
    :type random_type: str
    :param custom_rand_gen: Random generator for 'custom' random_type.
                            Provide generator from variables:
                            custom_rand_gen(variables) = values
    :type custom_rand_gen: func
    :param log_transform: If True then logarithm will be used incide for
                          parameters.
    :type log_transform: bool
    :param maximize: If True then optimization will maximize function.
    :type maximize: bool
    """
    def __init__(self, gen_size=10, n_elitism=2,
                 p_mutation=0.3, p_crossover=0.3, p_random=0.2,
                 mut_strength=0.2, const_mut_strength=1.1,
                 mut_rate=0.2, const_mut_rate=1.2, mut_attempts=2,
                 eps=1e-2, n_stuck_gen=100,
                 selection_type='roulette_wheel', selection_random=False,
                 mutation_type='gaussian', one_fifth_rule=True,
                 crossover_type='uniform', crossover_k=None,
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        # Simple checks
        assert isinstance(gen_size, int)
        assert isinstance(n_elitism, int)
        assert (n_elitism < gen_size)
        assert (p_mutation >= 0 and p_mutation <= 1)
        assert (p_crossover >= 0 and p_crossover <= 1)
        assert (p_random >= 0 and p_random <= 1)
        assert (mut_rate >= 0 and mut_rate <= 1)
        assert (mut_strength >= 0 and mut_strength <= 1)
        assert (const_mut_rate >= 1 and const_mut_rate <= 2)
        assert (const_mut_strength >= 1 and const_mut_strength <= 2)

        self.q_agent = GreedyQAgent()
        self.reward_calculator = DivReward()
        self.state_calculator = BetterCountPairedState()
        self.cur_state = None
        self.cur_action = None
        self.cur_reward = None
        self.reward_debug = None

        # todo call super
        GeneticAlgorithm.__init__(self, gen_size, n_elitism,
                 p_mutation, p_crossover, p_random,
                 mut_strength, const_mut_strength,
                 mut_rate, const_mut_rate, mut_attempts,
                 eps, n_stuck_gen,
                 selection_type, selection_random,
                 mutation_type, one_fifth_rule,
                 crossover_type, crossover_k,
                 random_type, custom_rand_gen,
                 log_transform, maximize)

    def selection(self, f, variables, X_gen, Y_gen=None,
                  selection_type='roulette_wheel', selection_random=False):
        """
        Perform selection in genetic algorithm.
        Selection could be of different types:

        * Roulette Wheel - the better fitness function is the higher chance
          to be selected for mutation and crossover for the individual is.
        * Rank - almost the same as Roulette Wheel but with rank insted
          fitness function. This means weight=1 for the best individual,
          weight=2 for the second best and so on.

        :param X_gen: previous generation of individuals.
        :param Y_gen: fitnesses of the previous generation. If `None` then
                         will be evaluated.
        :param selection_type: type of selection. Could be 'roulette_wheel' or
                               'rank'.
        :param selection_random: if True then number of mutants and crossover's
                                 offsprings in new generation will be binomial
                                 random variable.

        :returns: new generation and its fitnesses.
        """
        # Evaluate fitness if None
        if Y_gen is None:
            Y_gen = [f(x) for x in X_gen]
        # Sort by value of fitness
        X_gen, Y_gen = sort_by_other_list(X_gen, Y_gen, reverse=False)

        # Simple checks
        assert len(X_gen[0]) == len(variables)
        assert len(X_gen) == len(Y_gen)

        # Start selection procedure
        if selection_type == 'roulette_wheel':
            Y_gen = np.array(Y_gen)
            if (np.all(Y_gen == Y_gen[0]) or
                    not (np.all(Y_gen < 0) or np.all(Y_gen > 0))):
                p = [1 / len(Y_gen) for _ in Y_gen]
            else:
                is_not_inf = np.logical_not(np.isinf(Y_gen))
                if np.sum(is_not_inf) == 1:  # special case
                    p = [float(x) for x in is_not_inf]
                else:
                    p = Y_gen / np.sum(Y_gen[is_not_inf])
                    p[np.isinf(p)] = 1  # will be inversed to 0
                    p[np.isnan(p)] = 1  # will be inversed to 0
                    # We need to reverse probs as we have minimization problem
                    p = 1 - p
                    p /= np.sum(p)
        elif selection_type == 'rank':
            n = len(X_gen)
            p = np.arange(1, n + 1) / (n * (n - 1))
            p /= np.sum(p)
        else:
            raise ValueError(f"Unknown selection type: {selection_type}.")

        # Generate numbers for each operation
        if selection_random:
            n_mutants = np.random.binomial(self.gen_size, self.p_mutation)
            n_offsprings = np.random.binomial(self.gen_size, self.p_crossover)
            n_random_gen = np.random.binomial(self.gen_size, self.p_random)
        else:
            n_mutants = int(self.gen_size * self.p_mutation)
            n_offsprings = int(self.gen_size * self.p_crossover)
            n_random_gen = int(self.gen_size * self.p_random)

        # 1. Elitism
        new_X_gen = list(X_gen[:self.n_elitism])
        new_Y_gen = list(Y_gen[:self.n_elitism])

        # 2. Mutation
        # number_of_better is the total amount of improved individuals generated by mutation
        number_of_better = 0
        number_of_better_than_best = 0
        best_fitness = float('inf')
        best_fitness_ind = None
        for i in range(n_mutants):
            x_ind = np.random.choice(range(len(X_gen)), p=p)
            x = X_gen[x_ind]
            mutants = self.mutation(x, variables, self.mutation_type,
                                    self.one_fifth_rule, self.mut_attempts)
            fitness = [f(x_mut) for x_mut in mutants]

            # Calculate number of better individuals
            if self.q_agent.is_strict:
                number_of_better_than_best += sum(f < Y_gen[0] for f in fitness)
                number_of_better += sum(f < Y_gen[x_ind] for f in fitness)
            else:
                number_of_better_than_best += sum(f <= Y_gen[0] for f in fitness)
                number_of_better += sum(f <= Y_gen[x_ind] for f in fitness)

            #            print("Time of main part of mutation: " + str(t3 - t1))

            # Calculate best fitness
            cur_best = np.min(fitness)
            if cur_best <= best_fitness:
                best_fitness = cur_best
                best_fitness_ind = x_ind

            # Take best mutant
            new_Y_gen.append(np.min(fitness))
            new_X_gen.append(mutants[fitness.index(new_Y_gen[-1])])

            # One more check for weights.
            # If new x is better, then we would like to decrease weights of
            # parameters back as this change was good.
            if new_Y_gen[-1] < Y_gen[x_ind]:
                if isinstance(x, WeightedMetaArray):
                    new_X_gen[-1].weights = x.weights

        # 3. Crossover
        for i in range(n_offsprings):
            ind1, ind2 = np.random.choice(range(len(X_gen)), size=2, p=p)
            parent1, parent2 = X_gen[ind1], X_gen[ind2]
            x = self.crossover(parent1, parent2, variables,
                               self.crossover_type, self.crossover_k)
            new_X_gen.append(x)
            new_Y_gen.append(f(x))

        # 4. Random individuals
        for i in range(n_random_gen):
            x = WeightedMetaArray(self.randomize(variables, self.random_type,
                                                 self.custom_rand_gen))
            x.metadata = 'r'
            new_X_gen.append(x)
            new_Y_gen.append(f(x))

        # Sort by fitness
        new_X_gen, new_Y_gen = sort_by_other_list(new_X_gen, new_Y_gen,
                                                  reverse=False)

        # Update learning values
        self.cur_reward = self.reward_calculator.calculate(best_fitness, Y_gen[best_fitness_ind],
                                                           new_Y_gen.index(best_fitness) + 1)
        new_state = self.state_calculator.calculate(number_of_better, number_of_better_than_best)
        if self.cur_action is not None:
            self.q_agent.update_experience(self.cur_state, new_state, self.cur_action, self.cur_reward)
        self.cur_state = new_state
        self.cur_action = self.q_agent.choose_action(self.cur_state)
        self.reward_debug = str(best_fitness) + " " + str(Y_gen[best_fitness_ind]) + " " + \
                            str(new_Y_gen.index(best_fitness)) + " " + str(self.q_agent.q_map.get((self.cur_state, self.cur_action)))

        # Case when actions have the same q_values
        diff = Y_gen[best_fitness_ind] - best_fitness
        if self.cur_action == -1:
            condition = (diff >= self.eps) if self.q_agent.is_strict else ((diff <= self.eps) and (diff >= 0))
            if condition:
                self.cur_action = 0
            else:
                self.cur_action = 1

        # Return new generation
        new_X_gen = new_X_gen[:self.gen_size]
        new_Y_gen = new_Y_gen[:self.gen_size]

        return new_X_gen, new_Y_gen

    def _update_run_info(self, run_info, x_best, y_best, X, Y,
                         n_eval, gen_time=None, n_impr_gen=None,
                         maxiter=None, maxeval=None):
        """
        Updates fields of `run_info`after one iteration of GA.

        Fields of run_info like `cur_mut_rate`, `cur_mut_strength`, `gen_times`
        are updated. Also message, success and status from :meth:`.is_stopped`
        are recorded to `result` field.

        :param run_info: Run info to update.
        :param gen_time: Time of iteration.
        :param n_impr_gen: Number of iteration when improvement happened.
        :param maxiter: Maximum number of iterations.
        :param maxeval: Maximum number of evaluations.
        """
        super(GeneticAlgorithm, self)._update_run_info(run_info=run_info,
                                                       x_best=x_best,
                                                       y_best=y_best,
                                                       X=X,
                                                       Y=Y,
                                                       n_eval=n_eval)
        # Update mutation rates and strength
        if n_impr_gen is not None:
            run_info.n_impr_gen = n_impr_gen

            # Our n_iter was already increased so -1 is applied
            is_impr = (n_impr_gen == run_info.result.n_iter - 1)
            if self.one_fifth_rule:
                run_info.cur_mut_rate = self.update_by_hqea(
                    run_info.cur_mut_rate,
                    self.const_mut_rate
                )
                run_info.cur_mut_rate = max(0.01, min(0.5, run_info.cur_mut_rate))
            is_mut_best = False
            x_best = run_info.result.x
            if hasattr(x_best, 'weights') and len(x_best.metadata) > 0:
                is_mut_best = x_best.metadata[-1] == 'm'
            run_info.cur_mut_strength = self.update_by_hqea(
                run_info.cur_mut_strength,
                self.const_mut_strength,
                is_mut_best
            )
            run_info.cur_mut_strength = max(0.01, min(0.5, run_info.cur_mut_strength))

        # Save gen_time
        if gen_time is not None:
            run_info.gen_times.append(gen_time)

        # Save learning info
        run_info.cur_state = self.cur_state
        run_info.cur_reward = self.cur_reward
        run_info.cur_action = self.cur_action
        run_info.reward_debug = self.reward_debug

        # Create message and success status
        stoped, status, message = self.is_stopped(run_info.result.n_iter,
                                                  run_info.result.n_eval,
                                                  run_info.n_impr_gen,
                                                  maxiter,
                                                  maxeval,
                                                  ret_status=True)
        run_info.success = stoped
        run_info.result.status = status
        run_info.result.message = message
        return run_info

    def update_by_hqea(self, value, const, condition=True):
        """
        Updates ``value`` according to HQEA algorithm and 'one-fifth' rule and ``const``.

        :param value: Value to change.
        :param const: Const for one fifth rule.
        :param condition: Bool for mutation strength -- if fitness was improved by mutation.
        """
        if condition and (self.cur_action == 0):
            return value * const
        return value / (const) ** (0.25)


    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        """
        Write report about one generation in report file.

        :param run_info: Run info that should have at least the following
                         fields:
                         * `result` (:class:`gadma.optimizers.OptimizerResult`\
                         type) - current result,
                         * `gen_times` - list of iteration times.
        :param report_file: File to write report. If None then to stdout.

        :note: All values are reported as is, i.e. `X_gen`, `x_best` should be\
               already translated from log scale if optimization did so;\
               `Y_gen` and `y_best` must be already multiplied by -1 if we\
               have maximization instead of minimization.
        """
        n_gen = run_info.result.n_iter
        X_gen = run_info.result.X_out
        Y_gen = run_info.result.Y_out
        x_best = run_info.result.x
        y_best = run_info.result.y
        mean_time = np.mean(run_info.gen_times)

        print(f"Generation #{n_gen}.", file=stream)
        print("Current generation of solutions:", file=stream)
        print("N", "Value of fitness function", "Solution",
              file=stream, sep='\t')

        for i, (x, y) in enumerate(zip(X_gen, Y_gen)):
            # Use parent's report write function
            string = Optimizer._n_iter_string(
                n_iter=i,
                variables=variables,
                x=x,
                y=f'{y: 5f}',
            )
            print(string, file=stream)

        print(f"Current mean mutation rate:\t{run_info.cur_mut_rate: 3f}",
              file=stream)
        print(f"Current mean number of params to change during mutation:\t"
              f"{max(int(run_info.cur_mut_strength * len(variables)), 1): 3d}",
              file=stream)
        print("State: ", run_info.cur_state, file=stream)
        print("Action: ", run_info.cur_action, file=stream)
        print("Reward: ", run_info.cur_reward, file=stream)
        print("Reward debug: ", run_info.reward_debug, file=stream)
        print("\n--Best solution by value of fitness function--", file=stream)
        print("Value of fitness:", y_best, file=stream)
        print("Solution:", file=stream, end='')

        string = Optimizer._n_iter_string(
            n_iter='',
            variables=variables,
            x=x_best,
            y='',
        )
        print(string, file=stream)

        if mean_time is not None:
            print(f"\nMean time:\t{mean_time:.3f} sec.\n", file=stream)
        print("\n", file=stream)


register_global_optimizer('Genetic_algorithm_HQEA', GeneticAlgorithmHQEA)
