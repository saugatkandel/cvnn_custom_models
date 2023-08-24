import dataclasses as dt
import numpy as np
import dill


def full_summary(layer):

    # check if this layer has layers
    if hasattr(layer, "layers"):
        print("summary for " + layer.name)
        layer.summary()

        for l in layer.layers:
            full_summary(l)


@dt.dataclass
class HistoryTrainedModel:
    history: dict = dt.field(default_factory=dict, init=False)
    params: dict = dt.field(default_factory=dict, init=False)
    epoch: list = dt.field(default_factory=list, init=False)
    epochs_per_fit: list = dt.field(default_factory=list, init=False)

    def add(self, history_obj):
        for k, val in history_obj.history.items():
            self.history.setdefault(k, [])
            self.history[k].extend(val)

        for k, val in history_obj.params.items():
            self.params.setdefault(k, [])
            self.params[k].append(val)
        self.epochs_per_fit.append(history_obj.epoch)
        if len(self.epoch) > 0:
            self.epoch = np.concatenate((self.epoch, self.epoch[-1] + np.array(history_obj.epoch) + 1))
        else:
            self.epoch = np.array(history_obj.epoch)

    def save(self, filename):
        print("Saving history to", filename)
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return dill.load(f)
