"""Additive squared errors with one RMSE reduction per update/evaluation."""
import math

from libcity.executor.evaluate import ForecastLoss


class RootMeanSquaredLoss(ForecastLoss):
    """Emit mean-squared statistics; the loop applies the outer square root.

    Backward accumulates gradients of the sum of squared errors. Dividing once
    by 2*count*RMSE recovers the derivative of the combined effective batch,
    including masks and a short final group. No micro-batch graph is retained.
    """
    def __init__(self,bundle,config):
        if config['name'] not in ('rmse','masked_rmse'):
            raise ValueError('RootMeanSquaredLoss requires rmse or masked_rmse')
        super().__init__(config,bundle.scaler,bundle.metadata.output_dim)

    def aggregate(self,total,weight):
        return math.sqrt(total/weight) if weight else 0.

    def gradient_divisor(self,total,weight):
        root = self.aggregate(total,weight)
        # At exact fit the squared-error gradient is zero. A nonzero divisor
        # selects the finite zero subgradient while retaining optimizer state
        # updates (e.g. momentum/weight decay) and the usual step schedule.
        return 2*weight*root if root else weight
