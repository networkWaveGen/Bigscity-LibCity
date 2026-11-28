"""Overflow-safe inverse-sigmoid curriculum probability from explicit progress."""
import math


def teacher_probability(batches_seen,decay):
    exponent = batches_seen/decay-math.log(decay)
    return 1/(1+math.exp(exponent)) if exponent<700 else 0.
