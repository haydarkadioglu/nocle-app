/* app.js — Nocle frontend logic */

const App = (() => {
  // ── State ────────────────────────────────────────────────────────
  let fileDuration = 0;
  let procDuration = 0;
  let progressPoller = null;
  let activePlayer = null;   // 'original' | 'processed' | null

  // ── DOM refs ─────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  const statusDot  = $('status-dot');
  const statusText = $('status-text');
  const fileInfo   = $('file-info');
  const btnBrowse  = $('btn-browse');
  const btnProcess = $('btn-process');
  const progressWrap  = $('progress-wrap');
  const progressFill  = $('progress-fill');
  const progressLabel = $('progress-label');
  const players       = $('players');
  const procPlayer    = $('processed-player');
  const btnSave       = $('btn-save');
  const spectDiv      = $('spectrograms');

  // ── Helpers ──────────────────────────────────────────────────────
  function fmt(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  }

  function setStatus(msg, state /* 'ready' | 'working' | 'error' | '' */) {
    statusText.textContent = msg;
    statusDot.className = 'status-dot ' + (state || '');
  }

  function getOptions() {
    return {
      spectral_gate: $('f-spectral').checked,
      wiener:        $('f-wiener').checked,
      gaussian:      $('f-gaussian').checked,
      normalize:     $('f-normalize').checked,
      wiener_size:   parseFloat($('p-wiener').value),
      gaussian_sigma: parseFloat($('p-sigma').value),
    };
  }

  // ── Settings persistence ─────────────────────────────────────────
  const FILTER_IDS = ['f-spectral','f-wiener','f-gaussian','f-normalize','f-spectrograms'];
  const PARAM_IDS  = ['p-wiener','p-sigma'];

  function applySettings(s) {
    FILTER_IDS.forEach(id => { if (id in s) $(id).checked = s[id]; });
    PARAM_IDS.forEach(id  => { if (id in s) $(id).value  = s[id]; });
    if (s.accent_color) setAccentColor(s.accent_color);
  }

  function collectSettings() {
    const s = {};
    FILTER_IDS.forEach(id => s[id] = $(id).checked);
    PARAM_IDS.forEach(id  => s[id] = $(id).value);
    const activeDot = document.querySelector('.color-dot.active');
    if (activeDot) s.accent_color = activeDot.dataset.color;
    return s;
  }

  function saveSettings() {
    pywebview.api.save_settings(collectSettings());
  }

  // ── Accent Theme Switcher ───────────────────────────────────────
  const COLOR_HOVERS = {
    '#6366f1': '#4f46e5',
    '#06b6d4': '#0891b2',
    '#10b981': '#059669',
    '#f43f5e': '#e11d48',
    '#f59e0b': '#d97706',
  };

  function setAccentColor(color) {
    document.documentElement.style.setProperty('--accent', color);
    const hoverColor = COLOR_HOVERS[color] || color;
    document.documentElement.style.setProperty('--accent-h', hoverColor);

    document.querySelectorAll('.color-dot').forEach(dot => {
      dot.classList.toggle('active', dot.dataset.color === color);
    });
  }

  document.querySelectorAll('.color-dot').forEach(dot => {
    dot.addEventListener('click', () => {
      const color = dot.dataset.color;
      setAccentColor(color);
      saveSettings();
    });
  });

  // ── Gain Boost Slider ───────────────────────────────────────────
  const pGain = $('p-gain');
  const pGainVal = $('p-gain-val');
  if (pGain) {
    pGain.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value).toFixed(1);
      pGainVal.textContent = val + 'x';
      pywebview.api.set_playback_gain(val);
    });
  }

  // ── Tab navigation ───────────────────────────────────────────────
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      $('tab-' + btn.dataset.tab).classList.add('active');
      $('filters').style.display = btn.dataset.tab === 'main' ? '' : 'none';
    });
  });

  // ── Browse ───────────────────────────────────────────────────────
  btnBrowse.addEventListener('click', async () => {
    btnBrowse.disabled = true;
    const r = await pywebview.api.browse_file();
    btnBrowse.disabled = false;
    if (!r || r.cancelled || !r.success) return;

    fileDuration = r.duration;
    fileInfo.innerHTML = `
      <div class="file-name">${r.filename}</div>
      <div class="file-meta">${fmt(r.duration)} &nbsp;·&nbsp; 16 kHz mono</div>`;

    $('time-orig').textContent = `0:00 / ${fmt(fileDuration)}`;
    $('tl-orig').style.width = '0%';
    players.style.display = 'flex';
    btnProcess.disabled = false;
    setStatus('File loaded — ready to process', 'ready');
  });

  // ── Process ──────────────────────────────────────────────────────
  btnProcess.addEventListener('click', async () => {
    btnProcess.disabled = true;
    btnBrowse.disabled  = true;
    progressWrap.style.display = 'flex';
    progressFill.style.width   = '0%';
    progressLabel.textContent  = '0%';
    setStatus('Processing…', 'working');

    const r = await pywebview.api.process_audio(getOptions());
    if (!r.success) {
      setStatus('Error: ' + r.error, 'error');
      btnProcess.disabled = false;
      btnBrowse.disabled  = false;
      progressWrap.style.display = 'none';
      return;
    }

    // Poll progress
    progressPoller = setInterval(async () => {
      const p = await pywebview.api.get_progress();
      const pct = Math.round(p.progress * 100);
      progressFill.style.width  = pct + '%';
      progressLabel.textContent = pct + '%';
    }, 200);
  });

  // ── Called from Python when done ─────────────────────────────────
  function onProcessingDone(data) {
    clearInterval(progressPoller);
    progressFill.style.width  = '100%';
    progressLabel.textContent = '100%';
    setTimeout(() => { progressWrap.style.display = 'none'; }, 600);

    procDuration = data.duration;
    $('time-proc').textContent = `0:00 / ${fmt(procDuration)}`;
    $('tl-proc').style.width = '0%';

    procPlayer.style.opacity       = '1';
    procPlayer.style.pointerEvents = '';

    btnProcess.disabled = false;
    btnBrowse.disabled  = false;
    setStatus('Done', 'ready');

    if ($('f-spectrograms').checked) requestSpectrograms();
  }

  function onProcessingError(msg) {
    clearInterval(progressPoller);
    progressWrap.style.display = 'none';
    setStatus('Error: ' + msg, 'error');
    btnProcess.disabled = false;
    btnBrowse.disabled  = false;
  }

  // ── Playback ─────────────────────────────────────────────────────
  function setupPlayer(btnPlay, btnStop, type) {
    btnPlay.addEventListener('click', async () => {
      if (activePlayer === type) return;   // already playing this one
      await pywebview.api.stop_audio();
      activePlayer = type;

      // swap icon to pause symbol
      btnPlay.querySelector('svg').innerHTML =
        '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';

      await pywebview.api.play_audio(type);
    });

    btnStop.addEventListener('click', async () => {
      await pywebview.api.stop_audio();
      resetPlayerIcon(type);
      activePlayer = null;
    });
  }

  function resetPlayerIcon(type) {
    const btn = $(type === 'original' ? 'play-orig' : 'play-proc');
    btn.querySelector('svg').innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  }

  function onTimeUpdate(pos, total, type) {
    const pct = total > 0 ? (pos / total * 100).toFixed(1) : 0;
    if (type === 'original') {
      $('time-orig').textContent = `${fmt(pos)} / ${fmt(total)}`;
      $('tl-orig').style.width   = pct + '%';
    } else {
      $('time-proc').textContent = `${fmt(pos)} / ${fmt(total)}`;
      $('tl-proc').style.width   = pct + '%';
    }
  }

  function onPlaybackFinished() {
    const type = activePlayer;
    activePlayer = null;
    if (type) resetPlayerIcon(type);
  }

  setupPlayer($('play-orig'), $('stop-orig'), 'original');
  setupPlayer($('play-proc'), $('stop-proc'), 'processed');

  // ── A/B Swap ──────────────────────────────────────────────────────
  const btnAbSwap = $('btn-ab-swap');
  if (btnAbSwap) {
    btnAbSwap.addEventListener('click', async () => {
      const res = await pywebview.api.swap_audio_player();
      if (res && res.success) {
        if (res.player === 'processed') {
          resetPlayerIcon('original');
          $('play-proc').querySelector('svg').innerHTML =
            '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
          activePlayer = 'processed';
        } else if (res.player === 'original') {
          resetPlayerIcon('processed');
          $('play-orig').querySelector('svg').innerHTML =
            '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
          activePlayer = 'original';
        }
      }
    });
  }

  // ── Keyboard Shortcuts ───────────────────────────────────────────
  window.addEventListener('keydown', (e) => {
    // Ignore key shortcuts if focused inside an input or select
    const activeEl = document.activeElement;
    if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'SELECT')) {
      return;
    }

    if (e.code === 'Space') {
      e.preventDefault();
      if (activePlayer) {
        pywebview.api.stop_audio();
        resetPlayerIcon(activePlayer);
        activePlayer = null;
      } else if (procDuration > 0) {
        $('play-proc').click();
      } else if (fileDuration > 0) {
        $('play-orig').click();
      }
    } else if (e.code === 'KeyA') {
      e.preventDefault();
      if (btnAbSwap) btnAbSwap.click();
    } else if (e.code === 'KeyM') {
      e.preventDefault();
      btnToggleRt.click();
    } else if ((e.ctrlKey || e.metaKey) && e.code === 'KeyO') {
      e.preventDefault();
      btnBrowse.click();
    } else if ((e.ctrlKey || e.metaKey) && e.code === 'KeyS') {
      e.preventDefault();
      if (!btnSave.disabled) btnSave.click();
    }
  });

  // ── Save ─────────────────────────────────────────────────────────
  btnSave.addEventListener('click', async () => {
    btnSave.disabled = true;
    const r = await pywebview.api.save_audio();
    btnSave.disabled = false;
    if (r.success) setStatus('Saved to ' + r.path, 'ready');
    else if (!r.cancelled) setStatus('Save failed: ' + r.error, 'error');
  });

  // ── Spectrograms (placeholder — Python pushes base64) ────────────
  function requestSpectrograms() {
    spectDiv.style.display = 'block';
  }

  // Called from Python: App.setSpectrogram('original', '<base64>')
  function setSpectrogram(type, b64) {
    const img = $(type === 'original' ? 'spec-original' : 'spec-processed');
    img.src = 'data:image/png;base64,' + b64;
    img.style.display = 'block';
    spectDiv.style.display = 'block';
  }

  // ── Settings: save on any change ─────────────────────────────────
  [...FILTER_IDS, ...PARAM_IDS].forEach(id => {
    const el = $(id);
    el.addEventListener('change', saveSettings);
    el.addEventListener('blur',   saveSettings);
  });

  // ── Presets ──────────────────────────────────────────────────────
  const PRESETS = {
    custom:     null,
    voice:      { 'f-spectral': true,  'f-wiener': false, 'f-gaussian': false, 'f-normalize': true,  'p-wiener': 15, 'p-sigma': 2.0 },
    wind:       { 'f-spectral': true,  'f-wiener': true,  'f-gaussian': true,  'f-normalize': true,  'p-wiener': 15, 'p-sigma': 2.0 },
    ai_only:    { 'f-spectral': false, 'f-wiener': false, 'f-gaussian': false, 'f-normalize': true,  'p-wiener': 15, 'p-sigma': 2.0 },
    aggressive: { 'f-spectral': true,  'f-wiener': true,  'f-gaussian': true,  'f-normalize': true,  'p-wiener': 21, 'p-sigma': 3.0 },
  };

  $('f-preset').addEventListener('change', (e) => {
    const val = e.target.value;
    const p = PRESETS[val];
    if (p) {
      applySettings(p);
      saveSettings();
    }
  });

  // ── Drag and Drop File Support ────────────────────────────────────
  const fileCard = $('file-info').closest('.card');

  ['dragenter', 'dragover'].forEach(eventName => {
    fileCard.addEventListener(eventName, (e) => {
      e.preventDefault();
      fileCard.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    fileCard.addEventListener(eventName, (e) => {
      e.preventDefault();
      fileCard.classList.remove('drag-over');
    }, false);
  });

  fileCard.addEventListener('drop', async (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      const path = files[0].path;
      if (path) {
        const r = await pywebview.api.get_audio_info(path);
        if (r && r.success) {
          fileDuration = r.duration;
          fileInfo.innerHTML = `
            <div class="file-name">${r.filename}</div>
            <div class="file-meta">${fmt(r.duration)} &nbsp;·&nbsp; 16 kHz mono</div>`;
          $('time-orig').textContent = `0:00 / ${fmt(fileDuration)}`;
          $('tl-orig').style.width = '0%';
          players.style.display = 'flex';
          btnProcess.disabled = false;
          setStatus('File loaded — ready to process', 'ready');
        }
      }
    }
  });

  // ── Clickable Seek Timelines ──────────────────────────────────────
  document.querySelectorAll('.timeline').forEach(tl => {
    tl.addEventListener('click', async (e) => {
      const rect = tl.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clickX / rect.width));
      await pywebview.api.seek_audio(ratio);
    });
  });

  // ── Live VU Meter callback ───────────────────────────────────────
  function onRealtimeVu(peak) {
    const vuFill = $('rt-vu-fill');
    const vuText = $('rt-vu-text');
    if (!vuFill || !vuText) return;
    const pct = Math.min(100, Math.round(peak * 250));
    vuFill.style.width = pct + '%';
    vuText.textContent = pct + '%';
  }

  // ── Real-Time Mic Controls ──────────────────────────────────────────
  let isRtRunning = false;
  const btnToggleRt  = $('btn-toggle-rt');
  const selInput     = $('rt-input-device');
  const selOutput    = $('rt-output-device');
  const selLatency   = $('rt-latency-mode');
  const rtStatusDot  = $('rt-status-dot');
  const rtStatusText = $('rt-status-text');

  async function loadAudioDevices() {
    const res = await pywebview.api.get_audio_devices();
    if (!res || !res.success) return;

    selInput.innerHTML = '';
    selOutput.innerHTML = '';

    res.inputs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = d.name;
      if (d.index === res.default_input) opt.selected = true;
      selInput.appendChild(opt);
    });

    res.outputs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = d.name;
      if (d.name.toUpperCase().includes('CABLE')) opt.selected = true;
      selOutput.appendChild(opt);
    });

    // If default input wasn't explicitly selected, fall back to first non-mapper option
    if (!selInput.value && selInput.options.length > 0) {
      selInput.selectedIndex = 0;
    }
  }

  btnToggleRt.addEventListener('click', async () => {
    if (!isRtRunning) {
      btnToggleRt.disabled = true;
      const inIdx = selInput.value;
      const outIdx = selOutput.value;
      const bufSize = selLatency.value;

      const res = await pywebview.api.start_realtime_mic(inIdx, outIdx, bufSize);
      btnToggleRt.disabled = false;

      if (res && res.success) {
        isRtRunning = true;
        btnToggleRt.textContent = 'Stop Live Mic';
        btnToggleRt.className = 'btn btn-primary';
        rtStatusDot.className = 'status-dot ready';

        if ($('rt-auto-default').checked) {
          const defRes = await pywebview.api.set_windows_default_mic('CABLE Output');
          if (defRes && defRes.success) {
            rtStatusText.textContent = 'Live mic active — CABLE Output set as Windows default mic';
          } else {
            rtStatusText.textContent = 'Live mic active — routing audio (default mic switch failed)';
          }
        } else {
          rtStatusText.textContent = 'Live mic active — routing audio';
        }
      } else {
        rtStatusDot.className = 'status-dot error';
        rtStatusText.textContent = 'Failed to start live stream';
      }
    } else {
      btnToggleRt.disabled = true;
      await pywebview.api.stop_realtime_mic();

      if ($('rt-auto-default').checked && selInput.selectedIndex >= 0) {
        const physName = selInput.options[selInput.selectedIndex].text;
        await pywebview.api.set_windows_default_mic(physName);
      }

      btnToggleRt.disabled = false;
      isRtRunning = false;
      btnToggleRt.textContent = 'Start Live Mic';
      btnToggleRt.className = 'btn btn-accent';
      rtStatusDot.className = 'status-dot';
      rtStatusText.textContent = 'Live mic stopped (restored default mic)';
      onRealtimeVu(0);
    }
  });

  function onRealtimeError(msg) {
    isRtRunning = false;
    btnToggleRt.textContent = 'Start Live Mic';
    btnToggleRt.className = 'btn btn-accent';
    rtStatusDot.className = 'status-dot error';
    rtStatusText.textContent = 'Error: ' + msg;
    onRealtimeVu(0);
  }

  // ── Boot ─────────────────────────────────────────────────────────
  async function boot() {
    // Load saved settings
    const s = await pywebview.api.load_settings();
    if (s) applySettings(s);

    // Load audio devices for RT tab
    loadAudioDevices();

    // Load model
    setStatus('Loading model…', 'working');
    const r = await pywebview.api.load_model();
    if (r.success) setStatus('Ready', 'ready');
    else           setStatus('Model failed: ' + r.error, 'error');
  }

  // pywebview fires this event when the bridge is ready
  window.addEventListener('pywebviewready', boot);

  // Expose callbacks for Python → JS calls
  return { onProcessingDone, onProcessingError, onTimeUpdate, onPlaybackFinished, setSpectrogram, onRealtimeError, onRealtimeVu };
})();

