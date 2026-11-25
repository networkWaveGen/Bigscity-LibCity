"""Shared scalar-loss accounting for batches with different sample counts."""


class SampleWeightedMean:
    """Combine batch-mean losses by real sample count, not by batch count."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, loss, batch):
        fields = getattr(batch, 'data', batch)
        size = len(next(iter(fields.values())))
        self.total += float(loss.item()) * size
        self.count += size

    def mean(self):
        if not self.count:
            raise ValueError('Validation data must contain at least one sample.')
        return self.total / self.count
