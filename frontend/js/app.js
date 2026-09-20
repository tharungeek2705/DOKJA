/**
 * DOKJA Operations Dashboard Client Application
 * Handles WebSocket telemetry, camera switching, HUD toggles, and real-time analytics.
 */

window.activeCameraId = "CAM-01";
let camerasData = [];
let hudEnabled = true;
let currentView = "stream";
let telemetrySocket = null;
let observationsHistory = [];

// System Clock
function startSystemClock() {
  const clockEl = document.getElementById("system-clock");
  function tick() {
    const now = new Date();
    clockEl.textContent = now.toTimeString().split(" ")[0] + " UTC" + (now.getTimezoneOffset() > 0 ? "-" : "+") + Math.abs(now.getTimezoneOffset() / 60);
  }
  setInterval(tick, 1000);
  tick();
}

// WebSocket Connection
function connectTelemetryWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/ws/telemetry`;

  telemetrySocket = new WebSocket(wsUrl);

  telemetrySocket.onopen = () => {
    console.log("[DOKJA WS] Telemetry link established.");
  };

  telemetrySocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleTelemetryUpdate(data);
    } catch (e) {
      console.error("[DOKJA WS] Error parsing telemetry:", e);
    }
  };

  telemetrySocket.onclose = () => {
    console.warn("[DOKJA WS] Connection closed. Reconnecting in 2s...");
    setTimeout(connectTelemetryWebSocket, 2000);
  };

  telemetrySocket.onerror = (err) => {
    console.error("[DOKJA WS] Socket error:", err);
    telemetrySocket.close();
  };
}

// Process Real-time Payload
function handleTelemetryUpdate(payload) {
  if (payload.cameras && payload.cameras.length > 0) {
    camerasData = payload.cameras;
    renderCameraList(camerasData);
    document.getElementById("camera-count").textContent = camerasData.length;
    const networkCount = document.getElementById("network-node-count");
    if (networkCount) networkCount.textContent = String(camerasData.length).padStart(2, "0");
    if (typeof updateMapCameras === "function") {
      updateMapCameras(camerasData);
    }
  }

  // Update Telemetry for currently active camera
  const telem = payload.telemetry ? payload.telemetry[window.activeCameraId] : null;
  if (telem) {
    renderTelemetryMetrics(telem);
    fetchCameraForecast(window.activeCameraId);
  }

  // Update Active Tracks Inspector for currently active camera
  const tracks = payload.active_tracks ? payload.active_tracks[window.activeCameraId] || [] : [];
  renderTracksInspector(tracks);

  // Update Observation Timeline
  if (payload.recent_observations && payload.recent_observations.length > 0) {
    updateTimelineTable(payload.recent_observations);
  }
}

// Render Camera Matrix Cards
function renderCameraList(cameras) {
  const container = document.getElementById("camera-list-container");
  if (!container) return;

  container.innerHTML = cameras.map(cam => {
    const isSelected = (cam.id === window.activeCameraId);
    return `
      <div class="camera-card ${isSelected ? 'active' : ''}" onclick="selectCamera('${cam.id}')">
        <div class="camera-card-top">
          <span class="camera-id">${cam.id}</span>
          <span class="badge ${cam.status === 'active' ? 'live-badge' : ''}">${cam.status.toUpperCase()}</span>
        </div>
        <div class="camera-name">${cam.name}</div>
        <div class="camera-card-meta">
          <span>${cam.location}</span>
          <span class="font-mono text-cyan">${cam.active_vehicles_count || 0} VEHICLES</span>
        </div>
      </div>
    `;
  }).join("");
}

window.applyCameraSource = async function() {
  const sourceType = document.getElementById("camera-source-type")?.value || "synthetic";
  const sourceInput = document.getElementById("camera-source-input");
  const source = (sourceInput?.value || "synthetic").trim();
  if (!source) return;

  try {
    const res = await fetch(`/api/cameras/${window.activeCameraId}/source`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, source_type: sourceType })
    });

    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.detail || 'Source update failed');
    }

    const updatedCam = payload.camera;
    const camIndex = camerasData.findIndex(cam => cam.id === updatedCam.id);
    if (camIndex !== -1) {
      camerasData[camIndex] = updatedCam;
    }

    renderCameraList(camerasData);
    window.selectCamera(updatedCam.id);
    const statusText = document.getElementById("viewport-resolution-badge");
    if (statusText) {
      statusText.textContent = sourceType === 'video' ? 'VIDEO MODE' : sourceType === 'rtsp' ? 'RTSP MODE' : sourceType === 'webcam' ? 'WEBCAM MODE' : 'SIM MODE';
    }
  } catch (err) {
    console.error('[DOKJA] Camera source update failed:', err);
    alert('Unable to switch camera source. Check the source value and try again.');
  }
};

// Select Active Camera Node
window.selectCamera = function(camId) {
  window.activeCameraId = camId;
  const cam = camerasData.find(c => c.id === camId);

  if (cam) {
    document.getElementById("active-cam-id").textContent = cam.id;
    document.getElementById("active-cam-name").textContent = cam.name;
    document.getElementById("active-cam-coords").textContent = `${cam.latitude.toFixed(4)}° N, ${cam.longitude.toFixed(4)}° E`;

    const modeSelect = document.getElementById("camera-source-type");
    const sourceInput = document.getElementById("camera-source-input");
    if (modeSelect) modeSelect.value = cam.source_type || "synthetic";
    if (sourceInput) sourceInput.value = cam.source || "synthetic";
  }

  // Update Stream Source
  const streamImg = document.getElementById("primary-stream-img");
  if (streamImg) {
    const streamType = hudEnabled ? "live" : "raw";
    streamImg.src = `/api/stream/${camId}/${streamType}?t=${Date.now()}`;
  }

  renderCameraList(camerasData);
  if (typeof focusCameraOnMap === "function" && currentView === "map") {
    focusCameraOnMap(camId, camerasData);
  }
};

// Render Telemetry & Flow Gauges
function renderTelemetryMetrics(telem) {
  document.getElementById("val-total-vehicles").textContent = telem.total_active_vehicles;
  document.getElementById("val-flow-rate").textContent = telem.flow_rate_per_min;
  document.getElementById("val-avg-speed").textContent = Math.round(telem.avg_speed_px_s);

  // Density status & gauge bar
  const densityBadge = document.getElementById("density-badge");
  densityBadge.textContent = telem.density_status.toUpperCase();
  if (telem.density_status === "Light") {
    densityBadge.className = "badge text-green";
  } else if (telem.density_status === "Moderate") {
    densityBadge.className = "badge text-amber";
  } else {
    densityBadge.className = "badge text-crimson";
  }

  const pct = Math.round(telem.density_score * 100);
  document.getElementById("density-pct").textContent = `${pct}%`;
  document.getElementById("density-gauge-bar").style.width = `${pct}%`;

  // Vehicle class distributions
  const counts = telem.class_counts || {};
  const total = Math.max(1, telem.total_active_vehicles);

  ["car", "motorcycle", "bus", "truck"].forEach(cls => {
    const cnt = counts[cls] || 0;
    const countEl = document.getElementById(`count-${cls}`);
    const barEl = document.getElementById(`bar-${cls}`);
    if (countEl) countEl.textContent = cnt;
    if (barEl) barEl.style.width = `${Math.min(100, Math.round((cnt / total) * 100))}%`;
  });
}

async function fetchCameraForecast(cameraId) {
  try {
    const res = await fetch(`/api/analytics/forecast/${cameraId}`);
    if (!res.ok) return;
    const forecast = await res.json();
    const forecastStatus = document.getElementById("forecast-status");
    const forecastVehicles = document.getElementById("forecast-vehicles");
    const forecastConfidence = document.getElementById("forecast-confidence");
    const forecastReasons = document.getElementById("forecast-reasons");

    if (!forecastStatus || !forecastVehicles || !forecastConfidence || !forecastReasons) return;

    forecastStatus.textContent = forecast.forecast_status.toUpperCase();
    forecastVehicles.textContent = `${forecast.predicted_total_vehicles}`;
    forecastConfidence.textContent = `${Math.round(forecast.forecast_confidence * 100)}%`;
    forecastReasons.textContent = forecast.reasons.join(" ");

    const mapStatus = forecast.forecast_status.toLowerCase();
    if (mapStatus.includes("severe") || mapStatus.includes("heavy")) {
      forecastStatus.style.color = "var(--color-crimson)";
    } else if (mapStatus.includes("moderate")) {
      forecastStatus.style.color = "var(--color-amber)";
    } else {
      forecastStatus.style.color = "var(--color-green)";
    }
  } catch (err) {
    console.warn("Forecast fetch failed:", err);
  }
}

// Render Active Tracks in Inspector
function renderTracksInspector(tracks) {
  const container = document.getElementById("tracks-inspector-list");
  document.getElementById("active-tracks-count").textContent = tracks.length;

  if (!tracks || tracks.length === 0) {
    container.innerHTML = `<div class="empty-state">No persistent tracks detected</div>`;
    return;
  }

  container.innerHTML = tracks.map(trk => {
    const colorLabel = (trk.vehicle_color && trk.vehicle_color !== 'unknown') ? trk.vehicle_color.toUpperCase() + ' ' : '';
    const plateHtml = trk.plate_text ? `
      <div style="margin-top: 4px;">
        <span class="plate-badge">${trk.plate_text} [${Math.round(trk.plate_confidence * 100)}%]</span>
      </div>
    ` : '';

    return `
      <div class="track-item-card">
        <div class="track-item-top">
          <span class="track-badge">TRK-${trk.track_id}</span>
          <span class="track-class-pill pill-${trk.class_name}">${colorLabel}${trk.class_name} (${Math.round(trk.confidence * 100)}%)</span>
        </div>
        ${plateHtml}
        <div class="track-item-bottom font-mono" style="margin-top: 4px;">
          <span>SPD: ${Math.round(trk.speed_px_s)} px/s [${trk.heading_cardinal}]</span>
          <span>DWELL: ${trk.dwell_time_seconds.toFixed(1)}s</span>
        </div>
      </div>
    `;
  }).join("");
}

// Global active observations cache
window.activeObservations = {};
let isSearchActive = false;

// Update Bottom Chronological Timeline Table
function updateTimelineTable(newObs) {
  if (isSearchActive) return; // Don't override while user is actively searching

  const tbody = document.getElementById("timeline-table-body");
  if (!tbody) return;

  newObs.forEach(obs => {
    window.activeObservations[obs.observation_id] = obs;
    if (!observationsHistory.some(o => o.observation_id === obs.observation_id)) {
      observationsHistory.unshift(obs);
    }
  });

  if (observationsHistory.length > 50) {
    observationsHistory = observationsHistory.slice(0, 50);
  }

  document.getElementById("obs-count-label").textContent = `${observationsHistory.length} ENTRIES`;
  renderTableRows(observationsHistory);
}

function renderTableRows(records) {
  const tbody = document.getElementById("timeline-table-body");
  if (!tbody) return;

  tbody.innerHTML = records.map(obs => {
    window.activeObservations[obs.observation_id] = obs;
    const timeFormatted = obs.timestamp.replace("T", " ").split(".")[0];
    const plateHtml = obs.plate_text 
      ? `<span class="plate-badge">${obs.plate_text}</span>` 
      : `<span style="color:var(--text-muted)">--</span>`;
    const colorHtml = `<span class="badge" style="text-transform:uppercase;">${obs.vehicle_color || 'UNKNOWN'}</span>`;
    const confVal = obs.plate_text ? Math.round(obs.plate_confidence * 100) : Math.round(obs.confidence * 100);

    const hasCrops = (obs.vehicle_crop_url || obs.plate_crop_url);
    const dossierBtn = hasCrops
      ? `<button class="thumb-preview-btn font-mono" onclick="openEvidenceModal('${obs.observation_id}')">
           <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
           INSPECT
         </button>`
      : `<span class="status-pill active" style="display:inline-flex; padding:1px 6px;">VERIFIED</span>`;

    return `
      <tr>
        <td class="text-cyan">${obs.observation_id}</td>
        <td>${obs.camera_id}</td>
        <td>${timeFormatted}</td>
        <td>TRK-${obs.track_id}</td>
        <td style="text-transform: uppercase;">${obs.vehicle_type}</td>
        <td>${colorHtml}</td>
        <td>${plateHtml}</td>
        <td class="text-green">${confVal}%</td>
        <td>${obs.latitude.toFixed(4)}, ${obs.longitude.toFixed(4)}</td>
        <td>${dossierBtn}</td>
      </tr>
    `;
  }).join("");
}

// Plate Search & Investigation Filter
window.executePlateSearch = async function() {
  const input = document.getElementById("input-plate-search");
  const query = input.value.trim();
  if (!query) return;

  isSearchActive = true;
  document.getElementById("btn-reset-search").classList.remove("hidden");

  try {
    const normalized = query.toLowerCase().replace(/\s+/g, "_");
    const vehicleTypes = ["car", "two_wheeler", "motorcycle", "bus", "truck", "auto", "autorickshaw"];
    const colors = ["white", "black", "silver", "red", "blue", "yellow", "green", "orange"];
    const filter = vehicleTypes.includes(normalized) ? `vehicle_type=${encodeURIComponent(normalized)}`
      : colors.includes(normalized) ? `color=${encodeURIComponent(normalized)}`
      : `plate=${encodeURIComponent(query)}`;
    const res = await fetch(`/api/investigation/search?${filter}`);
    const data = await res.json();
    document.getElementById("obs-count-label").textContent = `${data.length} MATCHES // "${query.toUpperCase()}"`;
    renderTableRows(data);

    if (!vehicleTypes.includes(normalized) && !colors.includes(normalized)) {
      const routeRes = await fetch(`/api/investigation/trajectory?plate=${encodeURIComponent(query)}`);
      const routeData = await routeRes.json();
      if (typeof window.renderTrajectoryOnMap === "function") {
        window.renderTrajectoryOnMap(routeData.geojson || { features: [] });
      }
    }
  } catch (err) {
    console.error("Plate search error:", err);
  }
};

window.resetPlateSearch = function() {
  isSearchActive = false;
  document.getElementById("input-plate-search").value = "";
  document.getElementById("btn-reset-search").classList.add("hidden");
  document.getElementById("obs-count-label").textContent = `${observationsHistory.length} ENTRIES`;
  renderTableRows(observationsHistory);
  if (typeof window.clearRouteLayers === "function") {
    window.clearRouteLayers();
  }
};

// Evidence Modal Controller
window.openEvidenceModal = function(obsId) {
  const obs = window.activeObservations[obsId];
  if (!obs) return;

  const modal = document.getElementById("evidence-modal");
  document.getElementById("modal-obs-title").textContent = `EVIDENCE DOSSIER // ${obs.observation_id}`;

  const vehImg = document.getElementById("modal-veh-img");
  const plateImg = document.getElementById("modal-plate-img");

  vehImg.src = obs.vehicle_crop_url || "/static/img/placeholder.png";
  plateImg.src = obs.plate_crop_url || "/static/img/placeholder.png";

  const grid = document.getElementById("modal-telemetry-grid");
  grid.innerHTML = `
    <div class="modal-stat-item"><span class="modal-stat-label">CAMERA JUNCTION</span><span class="modal-stat-val text-cyan">${obs.camera_id}</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">RECORD TIMESTAMP</span><span class="modal-stat-val">${obs.timestamp.replace('T', ' ').split('.')[0]}</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">TRACK ID</span><span class="modal-stat-val">TRK-${obs.track_id}</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">CLASSIFICATION</span><span class="modal-stat-val" style="text-transform:uppercase;">${obs.vehicle_color || ''} ${obs.vehicle_type}</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">RECOGNIZED PLATE</span><span class="modal-stat-val text-cyan">${obs.plate_text || 'UNRESOLVED'}</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">ANPR CONFIDENCE</span><span class="modal-stat-val text-green">${Math.round((obs.plate_confidence || obs.confidence) * 100)}%</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">ESTIMATED VELOCITY</span><span class="modal-stat-val text-amber">${Math.round(obs.speed_px_s || 0)} px/s</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">HEADING</span><span class="modal-stat-val">${obs.heading || 'N'}</span></div>
    <div class="modal-stat-item"><span class="modal-stat-label">GEOLOCATION</span><span class="modal-stat-val">${obs.latitude.toFixed(4)}, ${obs.longitude.toFixed(4)}</span></div>
  `;

  modal.classList.remove("hidden");
};

window.closeEvidenceModal = function(event) {
  if (event && event.target && event.target.closest && event.target.closest(".modal-card") && event.target.tagName !== "BUTTON") {
    return;
  }
  const modal = document.getElementById("evidence-modal");
  if (modal) modal.classList.add("hidden");
};

// Toggle HUD Overlay
window.toggleHud = function() {
  hudEnabled = !hudEnabled;
  const btn = document.getElementById("btn-toggle-hud");
  const indicator = btn.querySelector(".toggle-indicator");

  if (hudEnabled) {
    indicator.classList.add("on");
  } else {
    indicator.classList.remove("on");
  }

  // Refresh stream with updated mode
  window.selectCamera(window.activeCameraId);
};

// Switch View between Stream and GIS Map
window.switchMainView = function(view) {
  currentView = view;
  const streamWrapper = document.getElementById("stream-wrapper");
  const mapWrapper = document.getElementById("map-wrapper");
  const btnStream = document.getElementById("btn-view-stream");
  const btnMap = document.getElementById("btn-view-map");

  if (view === "stream") {
    streamWrapper.classList.remove("hidden");
    mapWrapper.classList.add("hidden");
    btnStream.classList.add("active");
    btnMap.classList.remove("active");
  } else {
    streamWrapper.classList.add("hidden");
    mapWrapper.classList.remove("hidden");
    btnStream.classList.remove("active");
    btnMap.classList.add("active");
    if (typeof initGisMap === "function") {
      initGisMap();
      if (typeof focusCameraOnMap === "function") {
        focusCameraOnMap(window.activeCameraId, camerasData);
      }
    }
  }
};

// Fullscreen
window.toggleFullscreen = function() {
  const container = document.getElementById("viewport-container");
  if (!document.fullscreenElement) {
    container.requestFullscreen().catch(err => console.error(err));
  } else {
    document.exitFullscreen();
  }
};

window.clearLocalTimeline = function() {
  observationsHistory = [];
  document.getElementById("timeline-table-body").innerHTML = "";
  document.getElementById("obs-count-label").textContent = "0 ENTRIES";
};

// Initialization on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  startSystemClock();
  connectTelemetryWebSocket();

  const enrollForm = document.getElementById("enroll-form");
  if (enrollForm) {
    enrollForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(enrollForm);
      const message = document.getElementById("enroll-message");
      const payload = Object.fromEntries(form.entries());
      payload.latitude = Number(payload.latitude);
      payload.longitude = Number(payload.longitude);
      payload.source_type = "rtsp";
      if (message) message.textContent = "Connecting camera…";
      try {
        const response = await fetch("/api/cameras", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Camera could not be connected.");
        if (message) message.textContent = "Camera connected.";
        enrollForm.reset();
        setTimeout(() => window.closeEnrollDialog(), 700);
      } catch (error) {
        if (message) message.textContent = error.message;
      }
    });
  }
});

window.openEnrollDialog = function() {
  document.getElementById("enroll-modal")?.classList.remove("hidden");
};

window.closeEnrollDialog = function(event) {
  if (event && event.target?.closest?.(".enroll-card")) return;
  document.getElementById("enroll-modal")?.classList.add("hidden");
};
