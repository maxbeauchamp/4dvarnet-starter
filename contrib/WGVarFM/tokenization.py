"""Point observations -> WeatherGenerator tokens, with HEALPix cell masking.

Data-agnostic: a stream is a set of points (lat, lon, values, datetimes) at its
own native resolution. Tokenization and token/target layouts are the
vendored WeatherGenerator helpers; this module only wires them together.
"""

import numpy as np
import torch

from contrib.WGVarFM.weathergen_ext._compat import IOReaderData
from contrib.WGVarFM.weathergen_ext.datasets.tokenizer import Tokenizer
from contrib.WGVarFM.weathergen_ext.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
    tokenize_space,
)

# Per-point source token channels: stream_id (1) + time encoding (5) + local coords (2),
# followed by the stream's data channels (no geoinfos).
SOURCE_META_CHANNELS = 1 + 5 + 2
# Per-point target coordinate encoding: stream_id (1) + time (5) + local vertex/centre coords (99).
TARGET_COORDS_SIZE = 1 + 5 + 99
# Targets are not split into fixed-size tokens; any size larger than a cell's point count works.
_TARGET_TOKEN_SIZE = 1 << 30


def source_size(num_channels: int) -> int:
    return SOURCE_META_CHANNELS + num_channels


class WGTokenizer(Tokenizer):
    def __init__(self, healpix_level: int):
        super().__init__(healpix_level)
        self.num_cells = 12 * 4**healpix_level

    def _rdata(self, lat, lon, values, datetimes):
        coords = torch.stack([torch.as_tensor(lat), torch.as_tensor(lon)], -1).to(torch.float32)
        values = torch.as_tensor(values, dtype=torch.float32)
        geoinfos = torch.zeros((len(values), 0), dtype=torch.float32)
        return IOReaderData(coords, geoinfos, values, np.asarray(datetimes))

    def cells_with_data(self, idxs_cells_lens) -> np.ndarray:
        return np.array([len(lens) > 0 for lens in idxs_cells_lens])

    @staticmethod
    def _token_mask(idxs_cells_lens, cell_keep: np.ndarray) -> np.ndarray:
        """Cell-level keep flags -> token-level mask (as WG's TokenizerMasking.cell_to_token_mask)."""
        flags = [np.full(len(lens), keep) for lens, keep in zip(idxs_cells_lens, cell_keep) if lens]
        return np.concatenate(flags) if flags else np.zeros(0, dtype=bool)

    def index(self, lat, lon, values, datetimes, token_size: int | None):
        """HEALPix cell index of the points. token_size=None -> target indexing (no padding)."""
        rdata = self._rdata(lat, lon, values, datetimes)
        if len(rdata.data) == 0:
            return rdata, None
        if token_size is None:
            idx = tokenize_space(rdata, _TARGET_TOKEN_SIZE, self.hl_target, pad_tokens=False)
        else:
            idx = tokenize_space(rdata, token_size, self.hl_source, pad_tokens=True)
        return rdata, idx

    def source(self, stream_id, token_size, rdata, idx, time_win, cell_keep):
        """Tokens of the kept cells: ((num_tokens, token_size, C), tokens per cell)."""
        num_ch = source_size(rdata.data.shape[-1])
        empty = (torch.zeros((0, token_size, num_ch)), torch.zeros(self.num_cells, dtype=torch.int32))
        if idx is None:
            return empty
        idxs_cells, idxs_cells_lens = idx
        mask_tokens = self._token_mask(idxs_cells_lens, cell_keep)
        if not mask_tokens.any():
            return empty
        tokens_cells, tokens_per_cell = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            None,
            stream_id,
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
        )
        return torch.stack(tokens_cells), tokens_per_cell.to(torch.int32)

    def target(self, stream_id, rdata, idx, time_win, cell_keep):
        """Target points of the kept cells, ordered by cell.

        Returns (coords encoding (N, TARGET_COORDS_SIZE), points per cell, values (N, C)).
        """
        empty = (
            torch.zeros((0, TARGET_COORDS_SIZE)),
            torch.zeros(self.num_cells, dtype=torch.int32),
            torch.zeros((0, rdata.data.shape[-1])),
        )
        if idx is None:
            return empty
        idxs_cells, idxs_cells_lens = idx
        mask_tokens = self._token_mask(idxs_cells_lens, cell_keep)
        if not mask_tokens.any():
            return empty
        data, _, _, coords_local, points_per_cell = tokenize_apply_mask_target(
            stream_id,
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            None,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
        )
        return coords_local, points_per_cell.to(torch.int32), data
