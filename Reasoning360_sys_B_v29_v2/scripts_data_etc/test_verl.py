import os
import pathlib, sys
import importlib

'''
try:
    verl = importlib.import_module("verl")
except ModuleNotFoundError:
    print("VERL is not installed in this env"); sys.exit(1)

root = pathlib.Path(verl.__file__).parent
cfg_dir_pkg = root / "trainer" / "config"
print("VERL package dir:", root)
print("Config dir exists? ", cfg_dir_pkg.exists())
if cfg_dir_pkg.exists():
    files = sorted(p.name for p in cfg_dir_pkg.rglob("*.yaml"))
    print("YAMLs under trainer/config (first 30):", files[:30])

'''
ACC_W = float(os.environ.get("ACC_W", "0.7"))
Z3_W = float(os.environ.get("Z3_W", "0.3"))

print(ACC_W
      )
print(Z3_W)
