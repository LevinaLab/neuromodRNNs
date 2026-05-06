"""Alignment experiment entry point for cue_accumulation."""
 
from src.train.train_cue_accumulation import SPEC
from src.modRNN.training.alignment import train_and_compute_alignment
 
 
def train_and_evaluate_entry(cfg):
    return train_and_compute_alignment(cfg, SPEC)
 