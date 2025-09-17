
cfg=$1
PY_ARGS=${@:2}
python examples/shapenetpart/main.py --cfg $cfg ${PY_ARGS}
