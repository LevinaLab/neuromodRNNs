"""Alignment experiment entry point for pattern_generation."""
 
from src.train.train_pattern_generation import SPEC
from src.modRNN.training.alignment import train_and_compute_alignment
 
 
def train_and_evaluate_entry(cfg):
    return train_and_compute_alignment(cfg, SPEC)