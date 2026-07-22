"""
setup_api.py — Python bridge for the first-run setup window.
"""
import ctypes
import os
import queue
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile

import sounddevice as sd
import webview


class SetupApi:
    def __init__(self, on_done_callback):
        self.window = None
        self._on_done = on_done_callback          # called when setup finishes
        self._install_path = os.path.join(tempfile.gettempdir(), 'nocle_setup')
        self._progress = 0.0
        self._status = 'idle'                      # idle|downloading|extracting|installing|done|error|reboot_required

    def set_window(self, window):
        self.window = window

    # ── Deps info ─────────────────────────────────────────────────────────────

    def get_missing_deps(self):
        from setup_checker import get_missing_deps
        return get_missing_deps()

    def is_admin(self):
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    # ── Path ─────────────────────────────────────────────────────────────────

    def get_default_path(self):
        return self._install_path

    def browse_path(self):
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            path = result[0] if isinstance(result, (list, tuple)) else result
            self._install_path = path
            return {'success': True, 'path': path}
        return {'success': False, 'cancelled': True}

    # ── Progress polling ──────────────────────────────────────────────────────

    def get_progress(self):
        return {'progress': self._progress, 'status': self._status}

    # ── Install VB-Audio ──────────────────────────────────────────────────────

    def install_vb_cable(self):
        threading.Thread(target=self._install_task, daemon=True).start()
        return {'success': True}

    def _install_task(self):
        try:
            from setup_checker import check_vb_cable, check_nircmd
            os.makedirs(self._install_path, exist_ok=True)

            # 0 — NirCmd download if needed
            if not check_nircmd():
                self._status = 'downloading'
                self._progress = 0.02
                nir_url = 'https://www.nirsoft.net/utils/nircmd-x64.zip'
                nir_zip = os.path.join(self._install_path, 'nircmd.zip')
                urllib.request.urlretrieve(nir_url, nir_zip)
                with zipfile.ZipFile(nir_zip, 'r') as z:
                    z.extractall('.')

            # 1 — VB Cable download if needed
            if not check_vb_cable():
                url      = 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip'
                zip_path = os.path.join(self._install_path, 'VBCABLE.zip')
                ext_path = os.path.join(self._install_path, 'VBCABLE')

                self._status = 'downloading'
                self._progress = 0.05

                def _hook(count, block, total):
                    if total > 0:
                        self._progress = 0.05 + 0.50 * min(count * block / total, 1.0)

                urllib.request.urlretrieve(url, zip_path, _hook)

                # Extract
                self._status = 'extracting'
                self._progress = 0.58
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(ext_path)
                self._progress = 0.65

                # Find installer
                setup_exe = self._find_exe(ext_path)
                if not setup_exe:
                    raise RuntimeError('Installer executable not found inside zip.')

                # Run (UAC prompt will appear)
                self._status = 'installing'
                self._progress = 0.68
                self.window.evaluate_js('Setup.onInstallerLaunched()')
                ret = ctypes.windll.shell32.ShellExecuteW(None, 'runas', setup_exe, None, None, 1)
                if ret <= 32:
                    raise RuntimeError(f'ShellExecute failed (code {ret}). Did you cancel the UAC prompt?')

                # Poll until device appears (max 60 s)
                self._status = 'waiting'
                for i in range(60):
                    time.sleep(1)
                    self._progress = min(0.68 + i * 0.005, 0.97)
                    try:
                        for d in sd.query_devices():
                            if 'CABLE' in d['name'].upper():
                                self._progress = 1.0
                                self._status = 'done'
                                self.window.evaluate_js('Setup.onInstallDone()')
                                return
                    except Exception:
                        pass

                # Device not found after 60 s → reboot probably needed
                self._status = 'reboot_required'
                self.window.evaluate_js('Setup.onRebootRequired()')
            else:
                self._progress = 1.0
                self._status = 'done'
                self.window.evaluate_js('Setup.onInstallDone()')

        except Exception as exc:
            self._status = 'error'
            msg = str(exc).replace('"', '\\"').replace('\n', ' ')
            self.window.evaluate_js(f'Setup.onInstallError("{msg}")')

    @staticmethod
    def _find_exe(folder):
        """Find setup exe in folder, prefer 64-bit."""
        found = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.exe'):
                    found.append(os.path.join(root, f))
        for p in found:
            if '64' in os.path.basename(p).lower():
                return p
        return found[0] if found else None

    # ── Finish setup ──────────────────────────────────────────────────────────

    def finish(self):
        """JS calls this when everything is ready → open main app."""
        self._on_done()
        self.window.destroy()
