from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor
import copy

class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopper
        :param patience: the maximum number of rounds tolerated
        :param verbose: whether to stop early
        :param delta: the regularization factor
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None

    def earlystop(self, loss, model=None):
        """
        :param loss: the loss score on validation set
        :param model: the model
        """
        value = -loss
        cv = loss

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStopper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            self.count = 0

# Tambahkan __all__ untuk ekspor eksplisit
__all__ = ['GraphAttnModel', 'load_lpa_subtensor', 'early_stopper']
