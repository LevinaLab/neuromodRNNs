"""
Schema for alignment-experiment configuration.
 
This is used only when `experiment_type` == "alignment". For standard
training, the field is present in the config tree (with defaults) but
unused.
"""
 
from dataclasses import dataclass, field
from typing import List
 
from omegaconf import MISSING
 
 
@dataclass
class ComparisonConfig:
    """
    One alignment comparison: which rule to compare against BPTT, plus
    any rule-specific kwargs.
 
    Each comparison gets its own time series in the output. The `tag`
    is the prefix used in output filenames, so it must be unique
    within the `comparisons` list of an AlignmentConfig.
 
    Examples:
        # Compare e_prop_hardcoded against BPTT.
        ComparisonConfig(tag="eprop", rule="e_prop_hardcoded")
 
        # Compare diffusion under three different modes against BPTT,
        # all in the same alignment run.
        ComparisonConfig(tag="diff_aligned",  rule="diffusion", diffusion_mode="aligned")
        ComparisonConfig(tag="diff_perstep",  rule="diffusion", diffusion_mode="shuffled_per_step")
        ComparisonConfig(tag="diff_fixed",    rule="diffusion", diffusion_mode="shuffled_fixed")
    """
 
    tag: str = MISSING            # filename prefix; must be unique within `comparisons`
    rule: str = MISSING           # learning rule name passed to compute_grads
    diffusion_mode: str = "aligned"   # only meaningful when rule == "diffusion"
 
 
@dataclass
class AlignmentConfig:
    """Settings for the alignment experiment."""
 
    # List of comparisons to run alongside the BPTT training. Each is
    # measured at the same intervals on the same training micro-batches.
    comparisons: List[ComparisonConfig] = field(
        default_factory=lambda: [
            ComparisonConfig(tag="eprop", rule="e_prop_hardcoded"),
            ComparisonConfig(
                tag="diffusion_aligned",
                rule="diffusion",
                diffusion_mode="aligned",
            ),
        ]
    )
 
    # Measure alignment every this many epochs. Probably 50 (matches
    # how the experiment was originally specified) but tunable.
    interval: int = 50
 