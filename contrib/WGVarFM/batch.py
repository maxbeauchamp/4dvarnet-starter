"""Minimal stand-in for WeatherGenerator's ModelBatch (one input step, one output step).

Exposes exactly what the vendored EncoderModule / Model.forward read.
"""

import torch


class WGStreamData:
    def __init__(self, source_tokens, target_coords, target_coords_lens, target_values):
        # lists indexed by input step / output step (single step here)
        self.source_tokens_cells = [source_tokens]
        self.target_coords = [target_coords]
        self.target_coords_lens = [target_coords_lens]
        self.target_values = [target_values]

    def to(self, device):
        for name in ("source_tokens_cells", "target_coords", "target_coords_lens", "target_values"):
            setattr(self, name, [t.to(device, non_blocking=True) for t in getattr(self, name)])
        return self


class WGSample:
    def __init__(self, streams_data: dict[str, WGStreamData], meta: dict):
        self.streams_data = streams_data
        self.meta = meta


class WGBatch:
    def __init__(self, samples: list[WGSample], tokens_lens: torch.Tensor):
        self.samples = samples
        # (num_steps_input, num_samples, num_streams, num_cells)
        self.tokens_lens = tokens_lens

    def to(self, device):
        for s in self.samples:
            for sd in s.streams_data.values():
                sd.to(device)
        self.tokens_lens = self.tokens_lens.to(device, non_blocking=True)
        return self

    def get_samples(self):
        return self.samples

    def get_num_source_steps(self):
        return self.tokens_lens.shape[0]

    def get_output_len(self):
        return 1

    def get_output_idxs(self):
        return [0]

    def get_device(self):
        return self.tokens_lens.device

    def __len__(self):
        return len(self.samples)


def collate_wg(items: list[dict], stream_names: list[str]) -> WGBatch:
    """items[i] = {"streams": {name: dict(source_tokens, source_tokens_lens, target_coords,
    target_coords_lens, target_values)}, "meta": {...}}"""
    samples, lens = [], []
    for item in items:
        streams = item["streams"]
        samples.append(
            WGSample(
                {
                    name: WGStreamData(
                        streams[name]["source_tokens"],
                        streams[name]["target_coords"],
                        streams[name]["target_coords_lens"],
                        streams[name]["target_values"],
                    )
                    for name in stream_names
                },
                item["meta"],
            )
        )
        lens.append(torch.stack([streams[name]["source_tokens_lens"] for name in stream_names]))
    tokens_lens = torch.stack(lens).unsqueeze(0).to(torch.int32)
    return WGBatch(samples, tokens_lens)
