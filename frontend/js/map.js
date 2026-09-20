/**
 * DOKJA GIS Tactical City Map Module
 * Leaflet-based geospatial intelligence layer.
 */
let mapInstance = null;
let cameraMarkers = {};
let routeLayers = [];

function clearRouteLayers() {
  if (!mapInstance) return;
  routeLayers.forEach(layer => {
    mapInstance.removeLayer(layer);
  });
  routeLayers = [];
}

function renderTrajectoryOnMap(geojson) {
  if (!mapInstance || !geojson || !geojson.features) {
    clearRouteLayers();
    return;
  }

  clearRouteLayers();

  geojson.features.forEach(feature => {
    if (!feature || feature.geometry?.type !== 'LineString') return;

    const latLngs = feature.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
    const polyline = L.polyline(latLngs, {
      color: '#00e5ff',
      weight: 4,
      opacity: 0.9,
      dashArray: '8 12'
    }).addTo(mapInstance);

    const pathPoints = latLngs.map(pos => L.circleMarker(pos, {
      radius: 4,
      color: '#ffb300',
      fillColor: '#ffb300',
      fillOpacity: 0.9,
      weight: 1
    }).addTo(mapInstance));

    routeLayers.push(polyline, ...pathPoints);
  });

  if (routeLayers.length > 0) {
    const routeGroup = L.featureGroup(routeLayers.filter(layer => layer instanceof L.Polyline || layer instanceof L.CircleMarker));
    if (routeGroup.getBounds && routeGroup.getBounds().isValid && routeGroup.getBounds().isValid()) {
      mapInstance.fitBounds(routeGroup.getBounds(), { padding: [24, 24] });
    }
  }
}

function initGisMap() {
  if (mapInstance) return;

  const mapContainer = document.getElementById('gis-map');
  if (!mapContainer) return;

  // Default centered on Tamil Nadu; selecting a camera zooms to its district.
  mapInstance = L.map('gis-map', {
    zoomControl: false,
    attributionControl: false
  }).setView([11.1271, 78.6569], 7);

  L.control.zoom({ position: 'bottomright' }).addTo(mapInstance);

  // CartoDB Dark Matter tactical map tiles
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    subdomains: 'abcd'
  }).addTo(mapInstance);
}

function updateMapCameras(cameras) {
  if (!mapInstance) return;

  cameras.forEach(cam => {
    const latLng = [cam.latitude, cam.longitude];
    const isSelected = (window.activeCameraId === cam.id);

    // Custom tactical marker HTML
    const markerHtml = `
      <div class="tactical-marker ${isSelected ? 'selected' : ''}" id="marker-${cam.id}">
        <div class="marker-pulse"></div>
        <div class="marker-core">
          <span class="marker-id">${cam.id}</span>
          <span class="marker-count">${cam.active_vehicles_count || 0}</span>
        </div>
      </div>
    `;

    const icon = L.divIcon({
      className: 'tactical-div-icon',
      html: markerHtml,
      iconSize: [44, 44],
      iconAnchor: [22, 22]
    });

    if (cameraMarkers[cam.id]) {
      cameraMarkers[cam.id].setLatLng(latLng);
      cameraMarkers[cam.id].setIcon(icon);
    } else {
      const marker = L.marker(latLng, { icon: icon }).addTo(mapInstance);
      marker.on('click', () => {
        if (typeof window.selectCamera === 'function') {
          window.selectCamera(cam.id);
        }
      });
      cameraMarkers[cam.id] = marker;
    }
  });
}

function focusCameraOnMap(camId, cameras) {
  if (!mapInstance) return;
  const targetCam = cameras.find(c => c.id === camId);
  if (targetCam) {
    mapInstance.flyTo([targetCam.latitude, targetCam.longitude], 14, {
      animate: true,
      duration: 1.2
    });
  }
}
