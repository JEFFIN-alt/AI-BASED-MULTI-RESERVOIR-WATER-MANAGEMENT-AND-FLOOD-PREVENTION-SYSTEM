export class TwinAPI {
  constructor(onStateUpdate) {
    this.onStateUpdate = onStateUpdate;
    this.ws = null;
    this.baseUrl = '/api';
    this.connect();
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/state`;
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        if (this.onStateUpdate) {
          this.onStateUpdate(state);
        }
      } catch (err) {
        console.error("Failed to parse state", err);
      }
    };
    
    this.ws.onclose = () => {
      console.log("WebSocket closed. Reconnecting...");
      setTimeout(() => this.connect(), 2000);
    };
  }

  async post(endpoint, data) {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      return await response.json();
    } catch (err) {
      console.error(`API Error on ${endpoint}:`, err);
    }
  }

  async setGate(reservoirId, value) {
    return this.post(`/gate/${reservoirId}`, { value: value / 100.0 });
  }

  async play() {
    return this.post('/simulation/play', {});
  }

  async pause() {
    return this.post('/simulation/pause', {});
  }

  async step() {
    return this.post('/simulation/step', {});
  }

  async reset() {
    return this.post('/simulation/reset', {});
  }

  async setSpeed(speed) {
    return this.post('/simulation/speed', { speed });
  }

  async setStorm(value) {
    return this.post('/storm', { value: value / 100.0 });
  }

  async setMode(mode) {
    return this.post('/controller/mode', { mode });
  }
}
