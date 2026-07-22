const Setup = (() => {
  const $ = id => document.getElementById(id);

  let poller = null;

  async function init() {
    const defaultPath = await pywebview.api.get_default_path();
    $('path-input').value = defaultPath;

    const missing = await pywebview.api.get_missing_deps();
    const listEl = $('dep-list');
    listEl.innerHTML = '';

    if (!missing || missing.length === 0) {
      listEl.innerHTML = '<p style="color:var(--success);font-size:13px;">All dependencies are already installed!</p>';
      $('btn-install').style.display = 'none';
      $('btn-launch').style.display = 'inline-block';
      $('path-row').style.display = 'none';
      return;
    }

    missing.forEach(dep => {
      const card = document.createElement('div');
      card.className = 'dep-card';
      card.innerHTML = `
        <div class="dep-card-header">
          <span class="dep-title">${dep.name}</span>
          <span class="dep-size">${dep.size}</span>
        </div>
        <div class="dep-desc">${dep.description}</div>
      `;
      listEl.appendChild(card);
    });

    $('btn-skip').style.display = 'inline-block';
  }

  $('btn-browse-path').addEventListener('click', async () => {
    const res = await pywebview.api.browse_path();
    if (res && res.success) {
      $('path-input').value = res.path;
    }
  });

  $('btn-install').addEventListener('click', async () => {
    $('btn-install').disabled = true;
    $('btn-skip').style.display = 'none';
    $('progress-section').style.display = 'flex';
    $('uac-note').style.display = 'block';

    pywebview.api.install_vb_cable();

    poller = setInterval(async () => {
      const p = await pywebview.api.get_progress();
      const pct = Math.round(p.progress * 100);
      $('pfill').style.width = pct + '%';

      if (p.status === 'downloading') $('pmsg').textContent = `Downloading installer… ${pct}%`;
      else if (p.status === 'extracting') $('pmsg').textContent = 'Extracting files…';
      else if (p.status === 'installing') $('pmsg').textContent = 'Waiting for administrator confirmation…';
      else if (p.status === 'waiting') $('pmsg').textContent = 'Finalizing installation…';
    }, 250);
  });

  $('btn-skip').addEventListener('click', () => {
    pywebview.api.finish();
  });

  $('btn-launch').addEventListener('click', () => {
    pywebview.api.finish();
  });

  function onInstallerLaunched() {
    $('pmsg').textContent = 'Installer window opened. Please complete the setup dialog.';
  }

  function onInstallDone() {
    clearInterval(poller);
    $('pfill').style.width = '100%';
    $('pmsg').textContent = 'Installation successful!';
    $('uac-note').style.display = 'none';
    $('btn-install').style.display = 'none';
    $('btn-launch').style.display = 'inline-block';
  }

  function onRebootRequired() {
    clearInterval(poller);
    $('uac-note').style.display = 'none';
    $('msg-reboot').style.display = 'block';
    $('btn-install').style.display = 'none';
    $('btn-launch').style.display = 'inline-block';
  }

  function onInstallError(msg) {
    clearInterval(poller);
    $('progress-section').style.display = 'none';
    $('uac-note').style.display = 'none';
    $('msg-error').textContent = 'Error: ' + msg;
    $('msg-error').style.display = 'block';
    $('btn-install').disabled = false;
    $('btn-skip').style.display = 'inline-block';
  }

  window.addEventListener('pywebviewready', init);

  return { onInstallerLaunched, onInstallDone, onRebootRequired, onInstallError };
})();
