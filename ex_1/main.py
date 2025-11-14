import pickle
import numpy as np

# Load the game data
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)


from typing import Dict, List, Tuple, Optional
import numpy as np

class ERMExperiment:
    """Base class for ERM experiments with prophets and game data."""
    
    def __init__(self, game_data: Dict, prophet_data: Dict):
        self.game_data = game_data
        self.prophet_data = prophet_data
    
    def erm_select_prophet(self, games: np.ndarray, predictions: np.ndarray) -> int:
        """Returns index of prophet with lowest error on games.
        
        Args:
            games: array of shape (n,) with true outcomes
            predictions: array of shape (k, n) with k prophets' predictions
        
        Returns:
            int: index of prophet with minimum error

        q: what happens if tie? select first by default,
        """
        errors = (predictions != games).sum(axis=1)
        

        # possible: smaple from tied
        min_error = errors.min()
        # Handle ties: sample uniformly among best
        best_indices = np.where(errors == min_error)[0]
        return np.random.choice(best_indices)
    
    def evaluate_prophet(self, selected_prophet_idx: int,
                         prophet_indices: Optional[np.ndarray] = None) -> Dict:
        """Evaluate approximation and estimation error.
        
        Args:
            selected_prophet_idx: index of prophet selected by ERM
        
        Returns:
            dict with test_set_error, approximation_error, estimation_error
        """
        # Test set error
        test_predictions = self.prophet_data['test_set'][selected_prophet_idx]
        test_games = self.game_data['test_set']
        test_error = (test_predictions != test_games).mean()
        
        # Approximation error: best available true risk in hypothesis class
        if prophet_indices is None:
            best_true_risk = self.prophet_data['true_risk'].min()
        else:
            best_true_risk = self.prophet_data['true_risk'][prophet_indices].min()
        
        
        # Estimation error: difference between selected and best
        selected_true_risk = self.prophet_data['true_risk'][selected_prophet_idx]
        estimation_error = selected_true_risk - best_true_risk
        
        return {
            'test_set_error': test_error,
            'approximation_error': best_true_risk,
            'estimation_error': estimation_error,
            'selected_prophet_idx': selected_prophet_idx
        }
    
    def sample_prophets(self) -> np.ndarray:
        """Select which prophets to consider for this trial.
        
        Override in subclasses to sample subset.
        
        Returns:
            Array of prophet indices to use
        """
        return np.arange(len(self.prophet_data['true_risk']))
    
    def select_train_games(self, n_games: int) -> np.ndarray:
        """Sample training game indices.
        
        Args:
            n_games: number of games to sample
        
        Returns:
            Array of game indices
        """
        return np.random.choice(len(self.game_data['train_set']), n_games, replace=False)
    
    def run_trial(self, n_train_games: int, trial_num: int, verbose: bool = False) -> Dict:
        """Run a single trial: select games, run ERM, evaluate.
        
        Args:
            n_train_games: number of training games to use
            trial_num: current trial number
            verbose: whether to print debug info
        
        Returns:
            dict with evaluation results
        """
        #pdb.set_trace()
        # Select prophets for this trial (default: all)
        prophet_indices = self.sample_prophets()
        
        # Select training games
        train_indices = self.select_train_games(n_train_games)
        game_results = self.game_data['train_set'][train_indices]
        prophet_predictions = self.prophet_data['train_set'][prophet_indices][:, train_indices]
        
        # Run ERM
        selected_relative_idx = self.erm_select_prophet(game_results, prophet_predictions)
        selected_idx = prophet_indices[selected_relative_idx]
        
        # Evaluate
        result = self.evaluate_prophet(selected_idx, prophet_indices)
        
        if verbose:
            print(f"Trial {trial_num+1}: selected prophet {selected_idx}, "
                  f"test_err={result['test_set_error']:.3f}, "
                  f"approx_err={result['approximation_error']:.3f}, "
                  f"est_err={result['estimation_error']:.3f}")
        
        return result
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics from trial results.
        
        Override to add custom metrics.
        
        Args:
            results: list of dicts from run_trial
        
        Returns:
            dict with aggregate metrics
        """
        test_errors = [r['test_set_error'] for r in results]
        estimation_errors = [r['estimation_error'] for r in results]
        approximation_errors = [r['approximation_error'] for r in results]
        
        return {
            'mean_test_error': np.mean(test_errors),
            'best_model_count': sum(e == 0 for e in estimation_errors),
            'mean_approximation_error': np.mean(approximation_errors),
            'mean_estimation_error': np.mean(estimation_errors)
        }
    
    def run_experiment(self, n_trials: int, n_train_games: int, verbose: bool = False) -> Dict:
        """Run full experiment with multiple trials.
        
        Args:
            n_trials: number of trials to run
            n_train_games: number of training games per trial
            verbose: whether to print debug info
        
        Returns:
            dict with aggregate metrics
        """
        results = [self.run_trial(n_train_games, i, verbose) for i in range(n_trials)]
        return self.compute_metrics(results)
    
    # scenario 3 class:

class ERMExperimentWithin1Percent(ERMExperiment):
    """Extended experiment that tracks within-1% metric."""
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics including within-1% count.
        
        Args:
            results: list of dicts from run_trial
        
        Returns:
            dict with aggregate metrics including within_1_percent_count
        """

        # Get base metrics
        base_metrics = super().compute_metrics(results)
       
        # Add within-1% metric
        estimation_errors = [r['estimation_error'] for r in results]
        within_1_percent = sum(e <= 0.01 for e in estimation_errors)
        
        return {
            **base_metrics,
            'within_1_percent_count': within_1_percent
        }
        
        
# scenario 5 class
import matplotlib.pyplot as plt

class SchoolExperiment(ERMExperimentWithin1Percent):
    """Experiment that samples k prophets per trial."""

    def __init__(self, game_data: Dict, prophet_data: Dict, k: int):
        super().__init__(game_data, prophet_data)
        self.k = k

    def sample_prophets(self) -> np.ndarray:
        """Sample k prophets uniformly from available prophets.

        Override base class to sample only k prophets instead of all.

        Returns:
            Array of k prophet indices
        """
        n_prophets = len(self.prophet_data['true_risk'])
        return np.random.choice(n_prophets, self.k, replace=False)
    
    def run_grid_search(self, k_values: List[int], m_values: List[int], 
                       n_trials: int = 100) -> Dict[Tuple[int, int], Dict]:
        """Run experiments for all (k, m) combinations.
        
        Args:
            k_values: list of k values (number of prophets to sample)
            m_values: list of m values (number of training games)
            n_trials: number of trials per combination
        
        Returns:
            dict mapping (k, m) -> metrics dict
        """
        results = {}
        
        for k in k_values:
            self.k = k  # Update k for this set of experiments
            for m in m_values:
                print(f"\nRunning k={k}, m={m}...")
                metrics = self.run_experiment(n_trials=n_trials, n_train_games=m)
                results[(k, m)] = metrics
        
        return results
    


    def plot_results_table(self, results: Dict[Tuple[int, int], Dict], 
                          k_values: List[int], m_values: List[int]) -> plt.Figure:
        """Create matplotlib table visualization of results.
        
        Args:
            results: dict from run_grid_search
            k_values: list of k values
            m_values: list of m values
        
        Returns:
            matplotlib Figure object
        """
        # Create figure with 3 subplots (one for each metric)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Scenario 5: School of Prophets Results', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('mean_test_error', 'Average Test Error'),
            ('mean_approximation_error', 'Approximation Error'),
            ('mean_estimation_error', 'Estimation Error')
        ]
        
        for ax, (metric_key, metric_name) in zip(axes, metrics_to_plot):
            # Prepare data matrix
            data_matrix = np.zeros((len(k_values), len(m_values)))
            for i, k in enumerate(k_values):
                for j, m in enumerate(m_values):
                    data_matrix[i, j] = results[(k, m)][metric_key]
            
            # Create table
            cell_text = []
            for i, k in enumerate(k_values):
                row = [f'{data_matrix[i, j]:.4f}' for j in range(len(m_values))]
                cell_text.append(row)
            
            # Row and column labels
            row_labels = [f'k={k}' for k in k_values]
            col_labels = [f'm={m}' for m in m_values]
            
            # Create table
            table = ax.table(cellText=cell_text,
                           rowLabels=row_labels,
                           colLabels=col_labels,
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header
            for j in range(len(m_values)):
                table[(0, j)].set_facecolor('#4CAF50')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            # Style row labels
            for i in range(len(k_values)):
                table[(i+1, -1)].set_facecolor('#2196F3')
                table[(i+1, -1)].set_text_props(weight='bold', color='white')
            
            # Color cells by value (heatmap style)
            vmin, vmax = data_matrix.min(), data_matrix.max()
            for i in range(len(k_values)):
                for j in range(len(m_values)):
                    val = data_matrix[i, j]
                    # Normalize to [0, 1]
                    if vmax > vmin:
                        normalized = (val - vmin) / (vmax - vmin)
                    else:
                        normalized = 0.5
                    # Color: light (low) to dark (high)
                    table[(i+1, j)].set_facecolor(plt.cm.YlOrRd(normalized * 0.7))
            
            ax.set_title(metric_name, fontsize=12, fontweight='bold', pad=20)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    
class BiasComplexityExperiment:
    """Compare two hypothesis classes on bias-complexity tradeoff."""
    
    def __init__(self, game_data: Dict, hypothesis1_data: Dict, hypothesis2_data: Dict):
        self.game_data = game_data
        # Create two ERMExperiment instances
        self.h1_experiment = ERMExperiment(game_data, hypothesis1_data)
        self.h2_experiment = ERMExperiment(game_data, hypothesis2_data)
    
    def run_comparison(self, n_trials: int, n_train_games: int) -> Tuple[Dict, Dict]:
        """Run both hypothesis classes on same training sets.
        
        Args:
            n_trials: number of trials
            n_train_games: games per trial
        
        Returns:
            (h1_results, h2_results) tuple of result dicts with raw trial data
        """
        h1_results = []
        h2_results = []
        
        for trial_num in range(n_trials):
            # Sample training games ONCE per trial
            train_indices = self.h1_experiment.select_train_games(n_train_games)

            # Run H1 with these games
            h1_trial = self._run_trial_with_fixed_games(
                self.h1_experiment, train_indices
            )
            h1_results.append(h1_trial)

            # Run H2 with SAME games
            h2_trial = self._run_trial_with_fixed_games(
                self.h2_experiment, train_indices
            )
            h2_results.append(h2_trial)
        
        return h1_results, h2_results
    
    def _run_trial_with_fixed_games(self, experiment: ERMExperiment,
                                    train_indices: np.ndarray) -> Dict:
        """Run single trial with pre-selected training games.
        
        Args:
            experiment: ERMExperiment instance
            train_indices: pre-selected training game indices
            trial_num: trial number
        
        Returns:
            Trial result dict
        """
        # Get prophet indices (all prophets in this case)
        prophet_indices = experiment.sample_prophets()
        
        # Get training data
        game_results = experiment.game_data['train_set'][train_indices]
        prophet_predictions = experiment.prophet_data['train_set'][prophet_indices][:, train_indices]
        
        # Run ERM
        selected_idx = experiment.erm_select_prophet(game_results, prophet_predictions)
        
        # Evaluate
        return experiment.evaluate_prophet(selected_idx, prophet_indices)
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics (reuse from ERMExperiment)."""
        test_errors = [r['test_set_error'] for r in results]
        estimation_errors = [r['estimation_error'] for r in results]
        approximation_errors = [r['approximation_error'] for r in results]
        
        return {
            'mean_test_error': np.mean(test_errors),
            'mean_approximation_error': np.mean(approximation_errors),
            'mean_estimation_error': np.mean(estimation_errors),
            'estimation_errors_raw': estimation_errors  # Keep for histogram
        }
    
    def plot_comparison(self, h1_metrics: Dict, h2_metrics: Dict) -> plt.Figure:
        """Plot side-by-side histograms of estimation errors.
        
        Args:
            h1_metrics: metrics dict for hypothesis 1
            h2_metrics: metrics dict for hypothesis 2
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Scenario 6: Bias-Complexity Tradeoff - Estimation Error Distribution', 
                    fontsize=14, fontweight='bold')
        
        # Get estimation errors
        h1_est_errors = h1_metrics['estimation_errors_raw']
        h2_est_errors = h2_metrics['estimation_errors_raw']
        
        # Use same bins for both histograms (meaningful comparison)
        all_errors = h1_est_errors + h2_est_errors
        bins = np.linspace(min(all_errors), max(all_errors), 20)
        
        # H1 histogram
        axes[0].hist(h1_est_errors, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(h1_metrics['mean_estimation_error'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {h1_metrics['mean_estimation_error']:.4f}")
        axes[0].set_title('Hypothesis 1 (5 prophets, high bias)\nRisk ∈ [0.3, 0.6]', fontweight='bold')
        axes[0].set_xlabel('Estimation Error')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # H2 histogram
        axes[1].hist(h2_est_errors, bins=bins, alpha=0.7, color='coral', edgecolor='black')
        axes[1].axvline(h2_metrics['mean_estimation_error'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {h2_metrics['mean_estimation_error']:.4f}")
        axes[1].set_title('Hypothesis 2 (500 prophets, low bias)\nRisk ∈ [0.25, 0.6]', fontweight='bold')
        axes[1].set_xlabel('Estimation Error')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig

def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    np.random.seed(0)
    with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
        prophet_data = pickle.load(f)

    experiment = ERMExperiment(data, prophet_data)
    results = experiment.run_experiment(n_trials=100, n_train_games=1)

    # Print results in uniform format
    print("=" * 70)
    print("SCENARIO 1: Two Prophets, One Game")
    print("=" * 70)
    print(f"Average test error:        {results['mean_test_error']:.4f}")
    print(f"Approximation error:       {results['mean_approximation_error']:.4f}")
    print(f"Estimation error:          {results['mean_estimation_error']:.4f}")
    print(f"Best prophet chosen:       {results['best_model_count']}/100 times")
    print("=" * 70)


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    np.random.seed(0)
    with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
        prophet_data = pickle.load(f)

    experiment = ERMExperiment(data, prophet_data)
    results = experiment.run_experiment(n_trials=100, n_train_games=10)

    # Print results in uniform format
    print("=" * 70)
    print("SCENARIO 2: Two Prophets, Ten Games")
    print("=" * 70)
    print(f"Average test error:        {results['mean_test_error']:.4f}")
    print(f"Approximation error:       {results['mean_approximation_error']:.4f}")
    print(f"Estimation error:          {results['mean_estimation_error']:.4f}")
    print(f"Best prophet chosen:       {results['best_model_count']}/100 times")
    print("=" * 70)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    np.random.seed(0)
    with open('scenario_three_and_four_prophets.pkl', 'rb') as f:
        prophet_data = pickle.load(f)

    experiment = ERMExperimentWithin1Percent(data, prophet_data)
    results = experiment.run_experiment(n_trials=100, n_train_games=10)

    # Print results in uniform format
    print("=" * 70)
    print("SCENARIO 3: Many Prophets, Ten Games")
    print("=" * 70)
    print(f"Average test error:        {results['mean_test_error']:.4f}")
    print(f"Approximation error:       {results['mean_approximation_error']:.4f}")
    print(f"Estimation error:          {results['mean_estimation_error']:.4f}")
    print(f"Best prophet chosen:       {results['best_model_count']}/100 times")
    print(f"Within 1% of best:         {results['within_1_percent_count']}/100 times")
    print("=" * 70)


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    np.random.seed(0)
    with open('scenario_three_and_four_prophets.pkl', 'rb') as f:
        prophet_data = pickle.load(f)

    experiment = ERMExperimentWithin1Percent(data, prophet_data)
    results = experiment.run_experiment(n_trials=100, n_train_games=1000)

    # Print results in uniform format
    print("=" * 70)
    print("SCENARIO 4: Many Prophets, Many Games")
    print("=" * 70)
    print(f"Average test error:        {results['mean_test_error']:.4f}")
    print(f"Approximation error:       {results['mean_approximation_error']:.4f}")
    print(f"Estimation error:          {results['mean_estimation_error']:.4f}")
    print(f"Best prophet chosen:       {results['best_model_count']}/100 times")
    print(f"Within 1% of best:         {results['within_1_percent_count']}/100 times")
    print("=" * 70)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    np.random.seed(0)
    with open('scenario_five_prophets.pkl', 'rb') as f:
        prophet_data = pickle.load(f)

    k_values = [2, 5, 10, 50]
    m_values = [1, 10, 50, 1000]

    # Note: k is set dynamically in run_grid_search, initial value doesn't matter
    experiment = SchoolExperiment(data, prophet_data, k=2)
    results = experiment.run_grid_search(k_values, m_values, n_trials=100)

    # Print results in uniform format
    print("=" * 70)
    print("SCENARIO 5: School of Prophets")
    print("=" * 70)
    print(f"Grid search completed")
    print(f"  k values (prophets):     {k_values}")
    print(f"  m values (train games):  {m_values}")
    print(f"  Number of trials:        100")
    print(f"\nResults visualization saved to 'scenario_5_results.png'")
    print("=" * 70)

    # Generate and save visualization
    fig = experiment.plot_results_table(results, k_values, m_values)
    plt.savefig('scenario_5_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    np.random.seed(0)
    with open('scenario_six_prophets.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    hypothesis1 = data_dict['hypothesis1']
    hypothesis2 = data_dict['hypothesis2']

    # Run comparison
    experiment = BiasComplexityExperiment(data, hypothesis1, hypothesis2)
    h1_results, h2_results = experiment.run_comparison(n_trials=100, n_train_games=10)

    # Compute metrics
    h1_metrics = experiment.compute_metrics(h1_results)
    h2_metrics = experiment.compute_metrics(h2_results)

    # Print results in uniform format
    print("=" * 70)
    print("SCENARIO 6: Bias-Complexity Tradeoff")
    print("=" * 70)

    print("\nHypothesis 1 (5 prophets, high bias):")
    print(f"  Average test error:      {h1_metrics['mean_test_error']:.4f}")
    print(f"  Approximation error:     {h1_metrics['mean_approximation_error']:.4f}")
    print(f"  Estimation error:        {h1_metrics['mean_estimation_error']:.4f}")

    print("\nHypothesis 2 (500 prophets, low bias):")
    print(f"  Average test error:      {h2_metrics['mean_test_error']:.4f}")
    print(f"  Approximation error:     {h2_metrics['mean_approximation_error']:.4f}")
    print(f"  Estimation error:        {h2_metrics['mean_estimation_error']:.4f}")

    print(f"\nHistograms saved to 'scenario_6_histograms.png'")
    print("=" * 70)

    # Generate and save visualization
    fig = experiment.plot_comparison(h1_metrics, h2_metrics)
    plt.savefig('scenario_6_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    Scenario_1()
    Scenario_2()
    Scenario_3()
    Scenario_4()
    Scenario_5()
    Scenario_6()

