import pandas as pd
import seaborn as sns

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

class MetricLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()

        self.history = []

    @property
    def name(self):
        return "Logger_custom_plot"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        _metrics = {}
        for k, v in metrics.items():
            if 'train_' not in k and 'val_' not in k:
                _metrics[k] = v
            else:
                _split, _met = k.split('_')
                _metrics['split'] = _split
                _metrics[_met] = v
        
        _metrics['step'] = step
        self.history.append(_metrics)
        return

    def log_hyperparams(self, params):
        pass
    
    def plot(self, name, ax, **kwargs):
        self.df = pd.DataFrame(self.history)
        sns.lineplot(data=self.df, x='epoch', y='loss', hue='split', ax=ax, **kwargs)
        