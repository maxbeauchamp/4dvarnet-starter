from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import esm_parser

OmegaConf.register_new_resolver(
    "_singleton",
    lambda k: dict(
        _target_="main.SingletonStore.get",
        key=k,
        obj_cfg="${" + k + "}",
    ),
    replace=True,
)

OmegaConf.register_new_resolver(
    "singleton", lambda k: "${oc.create:${_singleton:" + k + "}}", replace=True
)


class SingletonStore:
    STORE = dict()

    @classmethod
    def get(cls, key, obj_cfg):
        return cls.STORE.setdefault(key, obj_cfg())

    @classmethod
    def clear(cls):
        cls.STORE = {}


cs = ConfigStore.instance()

domains = {
    "eNATL": dict(lon=[-100, 42], lat=[7, 69]),
    "ceNATL": dict(lon=[-61, -9], lat=[12, 64]),
    "NATL": dict(lon=[-77, 5], lat=[27, 64]),
    "cNATL": dict(lon=[-51, -9], lat=[32, 54]),
    "osmosis": dict(lon=[-22.5, -10.5], lat=[44, 56]),
    "gf": dict(lon=[-66, -54], lat=[32, 44]),
    "fgf": dict(lon=[-66, -54], lat=[33, 45]),
    "2gf": dict(lon=[-71., -49.], lat=[32, 44]),
    "4gf": dict(lon=[-71., -29.], lat=[32, 44]),
    "calm": dict(lon=[-41., -29.], lat=[32, 44]),
    "qnatl": dict(lon=[-77., 0.], lat=[27., 64.]),
    "canaries": dict(lon=[-31, -14], lat=[33, 46]),
    "canaries_t": dict(lon=[-29, -17], lat=[33, 45]),
    "baltic": dict(lon=[-10, 30], lat=[48, 66]),
    "baltic_ext": dict(lon=[-12, 32], lat=[46, 68.001]),
    "baltic_dm1_eval": dict(lon=[0.379999, 7.38005], lat=[53.98, 57.78001]),
    "baltic_dm2_eval": dict(lon=[-5.61999, 3.58005], lat=[61.18, 65.38001]),
    "baltic_dm3_eval": dict(lon=[16.93999, 20.68005], lat=[55.18, 56.8001]),
    "baltic_dm1": dict(lon=[1.48,6.28], lat=[53.48,58.28]),
    "baltic_dm2": dict(lon=[-6.32,3.28], lat=[60.88,65.68]),
    "baltic_dm3": dict(lon=[16.41,21.21], lat=[53.61,58.41]),
    "global": dict(lon=[-180,180], lat=[-80,90]),
    "arctic": dict(xc=[-3849750.,3749750.], yc=[5849750.,-5349750.]),
    "arctic_small": dict(xc=[-3849750.,3749750.], yc=[3849750.,-3349750.]),
    "arctic_tiny": dict(xc=[-3849750.,3749750.], yc=[449750.,-449750.]),
    "arctic_croscim": dict(xc=[-3849750.,3749750.], yc=[2473750.,-4896250.]),
}

for n, d in domains.items():
    key_coords = list(d.keys())
    train = {
        key_coords[0] : dict(_target_="builtins.slice", _args_=d[key_coords[0]]),
        key_coords[1] : dict(_target_="builtins.slice", _args_=d[key_coords[1]]),
    }
    test = {
        key_coords[0] : dict(_target_="builtins.slice", _args_=[d[key_coords[0]][0] + 1, d[key_coords[0]][1] - 1]),
        key_coords[1] : dict(_target_="builtins.slice", _args_=[d[key_coords[1]][0] + 1, d[key_coords[1]][1] - 1]),
    }
    cs.store(name=n, node={"train": train, "test": test}, group="domain")
