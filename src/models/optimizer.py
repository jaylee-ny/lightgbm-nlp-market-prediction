from typing import Dict, Any, Optional, Callable, List
import numpy as np
import pandas as pd
from loguru import logger

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed. Install with: pip install optuna")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from src.models.lgbm_model import LGBMModel


class BayesianOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna TPE sampler.
    
    Performs walk-forward cross-validation during optimization to ensure
    hyperparameters generalize to unseen data.
    
    Parameters
    ----------
    n_trials : int
        Number of optimization trials
    cv_folds : int
        Number of walk-forward CV folds
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 3,
        random_state: int = 42,
        n_jobs: int = 1
    ):
        if not HAS_OPTUNA:
            raise ImportError("Optuna required. Install with: pip install optuna")
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.study_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Optional[Dict[str, Any]] = None,
        objective_metric: str = 'log_loss',
        direction: str = 'minimize'
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training target
        param_space : Dict[str, Any], optional
            Custom parameter search space
        objective_metric : str
            Metric to optimize ('log_loss', 'accuracy', 'auc')
        direction : str
            'minimize' or 'maximize'
            
        Returns
        -------
        Dict[str, Any]
            Best hyperparameters found
        """
        logger.info(f"Starting Bayesian optimization with {self.n_trials} trials...")
        logger.info(f"Using {self.cv_folds}-fold walk-forward CV")
        
        if param_space is None:
            param_space = self._get_default_param_space()
        
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study_ = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        objective_fn = self._create_objective(
            X, y, param_space, objective_metric
        )
        
        self.study_.optimize(
            objective_fn,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best {objective_metric}: {self.best_score_:.6f}")
        logger.info(f"Best parameters: {self.best_params_}")
        
        return self.best_params_
    
    def _create_objective(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Dict[str, Any],
        metric: str
    ) -> Callable:
        """Create objective function for optimization."""
        
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, param_space)
            
            cv_scores = []
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train = X.iloc[train_idx]
                y_train = y[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y[val_idx]
                
                model = LGBMModel(
                    model_params=params,
                    early_stopping_rounds=50,
                    verbose=-1
                )
                
                model.fit(X_train, y_train, X_val, y_val)
                
                y_pred_proba = model.predict_proba(X_val)
                
                if metric == 'log_loss':
                    score = log_loss(y_val, y_pred_proba)
                elif metric == 'accuracy':
                    y_pred = model.predict(X_val)
                    score = (y_pred == y_val).mean()
                elif metric == 'auc':
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y_val, y_pred_proba[:, 1])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                cv_scores.append(score)
                
                trial.report(score, fold)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            self.optimization_history_.append({
                'trial': trial.number,
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })
            
            return mean_score
        
        return objective
    
    def _suggest_params(
        self,
        trial: optuna.Trial,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest parameters for trial."""
        params = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config['type']
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def _get_default_param_space(self) -> Dict[str, Any]:
        """
        Get default hyperparameter search space for LightGBM.
        
        Returns
        -------
        Dict[str, Any]
            Parameter search space configuration
        """
        return {
            'learning_rate': {
                'type': 'float',
                'low': 0.01,
                'high': 0.3,
                'log': True
            },
            'num_leaves': {
                'type': 'int',
                'low': 20,
                'high': 2000,
                'log': True
            },
            'min_data_in_leaf': {
                'type': 'int',
                'low': 50,
                'high': 1000,
                'log': True
            },
            'max_bin': {
                'type': 'int',
                'low': 127,
                'high': 511
            },
            'feature_fraction': {
                'type': 'float',
                'low': 0.6,
                'high': 1.0
            },
            'bagging_fraction': {
                'type': 'float',
                'low': 0.6,
                'high': 1.0
            },
            'bagging_freq': {
                'type': 'int',
                'low': 1,
                'high': 10
            },
            'min_gain_to_split': {
                'type': 'float',
                'low': 0.0,
                'high': 1.0
            }
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            History of all trials with scores
        """
        if not self.optimization_history_:
            return pd.DataFrame()
        
        history_records = []
        for entry in self.optimization_history_:
            record = {
                'trial': entry['trial'],
                'mean_score': entry['mean_score'],
                'std_score': entry['std_score']
            }
            record.update(entry['params'])
            history_records.append(record)
        
        return pd.DataFrame(history_records)
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
            return
        
        if self.study_ is None:
            logger.warning("No optimization study available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        history_df = self.get_optimization_history()
        
        axes[0].plot(history_df['trial'], history_df['mean_score'], marker='o', alpha=0.6)
        axes[0].axhline(self.best_score_, color='r', linestyle='--', label='Best')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('CV Score')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if hasattr(optuna.visualization, 'plot_param_importances'):
            try:
                from optuna.visualization.matplotlib import plot_param_importances
                plot_param_importances(self.study_, ax=axes[1])
            except Exception as e:
                logger.warning(f"Could not plot param importances: {e}")
                axes[1].text(0.5, 0.5, 'Param importance\nnot available',
                           ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig


class GridSearchOptimizer:
    """
    Simple grid search for comparison with Bayesian optimization.
    
    Useful for baseline comparison or when parameter space is small.
    """
    
    def __init__(self, cv_folds: int = 3):
        self.cv_folds = cv_folds
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = None
    
    def search(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        metric: str = 'log_loss'
    ) -> Dict[str, Any]:
        """
        Run grid search.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training target
        param_grid : Dict[str, List[Any]]
            Parameter grid to search
        metric : str
            Metric to optimize
            
        Returns
        -------
        Dict[str, Any]
            Best parameters found
        """
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = np.prod([len(v) for v in param_values])
        logger.info(f"Grid search over {total_combinations} parameter combinations")
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        best_score = float('inf') if metric == 'log_loss' else float('-inf')
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = LGBMModel(model_params=params, verbose=-1)
                model.fit(X_train, y_train)
                
                y_pred_proba = model.predict_proba(X_val)
                
                if metric == 'log_loss':
                    score = log_loss(y_val, y_pred_proba)
                else:
                    raise ValueError(f"Metric {metric} not supported")
                
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            
            self.results_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': np.std(cv_scores)
            })
            
            if (metric == 'log_loss' and mean_score < best_score) or \
               (metric != 'log_loss' and mean_score > best_score):
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = mean_score
        
        logger.info(f"Grid search complete!")
        logger.info(f"Best {metric}: {self.best_score_:.6f}")
        logger.info(f"Best parameters: {self.best_params_}")
        
        return self.best_params_
