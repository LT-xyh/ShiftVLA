import json
from pathlib import Path
from scripts.m1_sa_path_preflight import preflight

def test_missing_pinned_checkout_blocks():
    r = preflight()
    assert r['status'] == 'BLOCKED'
    assert {'lerobot_checkout','robosuite_checkout','mujoco_checkout'} <= set(r['missing_roles'])

def test_old_config_historical_path_is_visible():
    assert preflight()['unresolved_historical_path_in_old_config'] is True

def test_no_dynamic_modules_imported():
    import scripts.m1_sa_path_preflight
    assert 'mujoco' not in scripts.m1_sa_path_preflight.__dict__
