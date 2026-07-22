"""
setup_checker.py — Detects missing system dependencies for Nocle.
"""
import os
import sounddevice as sd


def check_vb_cable() -> bool:
    """Return True if VB-Audio Virtual Cable is detected as an audio device."""
    try:
        for d in sd.query_devices():
            name = d['name'].upper()
            if 'CABLE' in name or 'VB-AUDIO' in name:
                return True
    except Exception:
        pass
    return False


def check_nircmd() -> bool:
    """Return True if nircmd.exe exists in the project root."""
    return os.path.exists("nircmd.exe")


def get_missing_deps() -> list[dict]:
    """Return a list of dicts describing each missing dependency."""
    missing = []
    if not check_vb_cable():
        missing.append({
            'id':          'vb_cable',
            'name':        'VB-Audio Virtual Cable',
            'description': 'Virtual microphone driver — lets the app output processed audio as a mic source for games and chat apps.',
            'size':        '~3 MB',
            'url':         'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip',
        })
    if not check_nircmd():
        missing.append({
            'id':          'nircmd',
            'name':        'NirCmd Audio Helper',
            'description': 'Utility to automatically set the virtual microphone as your Windows default recording device.',
            'size':        '~120 KB',
            'url':         'https://www.nirsoft.net/utils/nircmd-x64.zip',
        })
    return missing


def all_deps_ok() -> bool:
    return len(get_missing_deps()) == 0

