"""
Dataloader for the GNN-based CROSCIM model.

The GNN model uses exactly the same TrainingItem namedtuple and batch format
as the UNet/CM models — it requires ``lat``, ``lon`` and ``land_mask`` fields
which are already part of every TrainingItem (see data_simple_multires_supervised.py).

This module is therefore a **thin re-export** of
:class:`BaseDataModuleMultiResSupervised_simplify`, providing a dedicated
entry-point so that the GNN config can reference
``contrib.CROSCIM.dataloaders.data_simple_multires_gnn.BaseDataModuleMultiResGNN``
without coupling it to the supervised-model file.

No structural changes to the dataset or collation are needed because:

* ``lat`` and ``lon`` are already stored as spatial arrays in the TrainingItem.
* ``land_mask`` is already stored in the TrainingItem.
* The GNN's graph construction (grid → graph → grid) happens entirely
  inside :class:`GNNSolver.forward`, not in the dataloader.
"""

from .data_simple_multires_supervised import (
    BaseDataModuleMultiResSupervised_simplify,
    XrDatasetMultiResSupervised_simplify,
    create_training_item,
    TrainingItem,
)

# Public alias used in the YAML config
BaseDataModuleMultiResGNN = BaseDataModuleMultiResSupervised_simplify

__all__ = [
    "BaseDataModuleMultiResGNN",
    "BaseDataModuleMultiResSupervised_simplify",
    "XrDatasetMultiResSupervised_simplify",
    "create_training_item",
    "TrainingItem",
]
