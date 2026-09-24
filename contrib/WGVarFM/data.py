"""Generic WeatherGenerator data pipeline: reader -> masked HEALPix tokens -> WGBatch.

A reader provides, per stream (= sensor) and per date, the valid observation points
at the stream's own resolution:
    reader.dates() -> sorted np.datetime64[D] dates where every stream has data
    reader.read(stream_name, date) -> dict(lat, lon, values (N, C), datetimes (N,))

Self-supervised masking (as in WG pre-training): for each sample and each stream that is
both source and target, a random `masking_rate` fraction of its non-empty HEALPix cells
is removed from the source and becomes the target.
"""

from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch

from contrib.WGVarFM.batch import collate_wg
from contrib.WGVarFM.tokenization import WGTokenizer


def _in_slices(dates, slices):
    slices = slices if isinstance(slices, (list, tuple)) else [slices]
    keep = np.zeros(len(dates), dtype=bool)
    for sl in slices:
        keep |= (dates >= np.datetime64(sl.start)) & (dates <= np.datetime64(sl.stop))
    return dates[keep]


class WGDataset(torch.utils.data.Dataset):
    def __init__(self, reader, dates, streams, healpix_level, masking_rate, deterministic, seed=0):
        self.reader = reader
        self.dates = dates
        self.streams = streams
        self.masking_rate = masking_rate
        self.deterministic = deterministic
        self.seed = seed
        self.tokenizer = WGTokenizer(healpix_level)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        time_win = (date, date + np.timedelta64(1, "D"))
        rng = np.random.default_rng(self.seed + idx if self.deterministic else None)
        num_cells = self.tokenizer.num_cells

        out = {}
        for stream_id, (name, sc) in enumerate(self.streams.items()):
            pts = self.reader.read(name, date)
            is_source, is_target = sc.get("source", True), sc.get("target", False)

            rdata, idx_src = self.tokenizer.index(
                pts["lat"], pts["lon"], pts["values"], pts["datetimes"], sc["token_size"]
            )
            has_data = (
                self.tokenizer.cells_with_data(idx_src[1])
                if idx_src is not None
                else np.zeros(num_cells, dtype=bool)
            )
            if is_source and is_target:
                masked = has_data & (rng.random(num_cells) < self.masking_rate)
            else:
                masked = has_data if is_target else np.zeros(num_cells, dtype=bool)

            src_keep = has_data & ~masked if is_source else np.zeros(num_cells, dtype=bool)
            src_tokens, src_lens = self.tokenizer.source(
                stream_id, sc["token_size"], rdata, idx_src, time_win, src_keep
            )

            _, idx_tgt = self.tokenizer.index(
                pts["lat"], pts["lon"], pts["values"], pts["datetimes"], None
            )
            tgt_coords, tgt_lens, tgt_values = self.tokenizer.target(
                stream_id, rdata, idx_tgt, time_win, masked
            )
            out[name] = dict(
                source_tokens=src_tokens,
                source_tokens_lens=src_lens,
                target_coords=tgt_coords,
                target_coords_lens=tgt_lens,
                target_values=tgt_values,
            )
        return {"streams": out, "meta": {"date": str(date)}}


class WGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        reader,
        streams,
        domains,
        healpix_level,
        masking_rate=0.5,
        batch_size=1,
        num_workers=0,
        seed=0,
    ):
        super().__init__()
        self.reader = reader
        self.streams = streams
        self.domains = domains
        self.healpix_level = healpix_level
        self.masking_rate = masking_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.collate = partial(collate_wg, stream_names=list(streams.keys()))
        self.datasets = {}

    def setup(self, stage=None):
        dates = self.reader.dates()
        for split in ("train", "val", "test"):
            if split in self.domains:
                self.datasets[split] = WGDataset(
                    self.reader,
                    _in_slices(dates, self.domains[split]["time"]),
                    self.streams,
                    self.healpix_level,
                    self.masking_rate,
                    deterministic=split != "train",
                    seed=self.seed,
                )

    def _loader(self, split, shuffle):
        return torch.utils.data.DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)
