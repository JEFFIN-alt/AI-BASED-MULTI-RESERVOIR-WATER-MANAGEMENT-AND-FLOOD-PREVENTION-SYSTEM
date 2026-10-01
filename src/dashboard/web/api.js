export class TwinAPI {
  constructor(onStateUpdate, onConnection = () => {}, onCommand = () => {}, onHealth = () => {}) {
    this.onStateUpdate = onStateUpdate;
    this.onConnection = onConnection;
    this.onCommand = onCommand;
    this.onHealth = onHealth;
    this.commandId = 0;
    this.ws = null;
    this.baseUrl = '/api';
    this.lastStateAt = 0;
    this.lastStateId = null;
    this.lastHeartbeatAt = 0;
    this.restStatus = 'CHECKING';
    this.websocketStatus = 'CONNECTING';
    this.stateStatus = 'WAITING';
    this.connect();
    this.checkFreshness();
    this.healthTimer = setInterval(() => this.checkFreshness(), 5000);
    this.healthTimer.unref?.();
  }

  publishHealth() {
    const wsOpen = this.ws?.readyState === WebSocket.OPEN;
    const connected = this.restStatus === 'LIVE' && this.websocketStatus === 'LIVE';
    const waiting = this.restStatus === 'CHECKING' || this.websocketStatus === 'CONNECTING'
      || this.websocketStatus === 'OPEN · AWAITING STATE';
    const status = connected ? (this.stateStatus === 'STALE' ? 'STALE DATA' : 'CONNECTED')
      : waiting ? 'CONNECTING'
      : (this.restStatus === 'LIVE' || this.websocketStatus === 'LIVE') ? 'STALE DATA' : 'DISCONNECTED';
    this.onHealth({rest: this.restStatus, websocket: this.websocketStatus,
      state: this.stateStatus, stateId: this.lastStateId,
      receivedAt: this.lastStateAt || null, status});
    this.onConnection(status);
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/state`);
    this.websocketStatus = 'CONNECTING';
    this.publishHealth();
    this.ws.onopen = () => {
      this.websocketStatus = 'OPEN · AWAITING STATE';
      this.publishHealth();
      this.checkFreshness();
    };
    this.ws.onerror = () => {
      this.websocketStatus = 'ERROR';
      this.publishHealth();
    };
    this.ws.onmessage = event => {
      try {
        const state = JSON.parse(event.data);
        if (state && state.type === 'heartbeat') {
          this.lastHeartbeatAt = Date.now();
          this.websocketStatus = 'LIVE';
          this.publishHealth();
          return;
        }
        if (!state || !state.state_identity) throw Error('State identity missing');
        this.lastStateAt = Date.now();
        this.lastHeartbeatAt = this.lastStateAt;
        this.lastStateId = state.state_identity.state_id;
        this.websocketStatus = 'LIVE';
        this.stateStatus = 'LIVE';
        if (this.onStateUpdate) this.onStateUpdate(state);
        this.publishHealth();
      } catch (err) { console.error('Invalid authoritative WebSocket state', err); }
    };
    this.ws.onclose = () => {
      this.websocketStatus = 'DISCONNECTED';
      this.stateStatus = 'STALE';
      this.publishHealth();
      setTimeout(() => this.connect(), 2000);
    };
  }

  async checkFreshness() {
    if (this.healthPending) return;
    this.healthPending = true;
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 4000);
    try {
      const response = await fetch(`${this.baseUrl}/state`, {signal: abort.signal, cache: 'no-store'});
      if (!response.ok) throw Error('state unavailable');
      const state = await response.json();
      this.restStatus = 'LIVE';
      const restId = state.state_identity?.state_id;
      const sameIdentity = restId != null && this.lastStateId != null && String(restId) === String(this.lastStateId);
      const grace = Math.max(10000, 2000 / (state.simulation?.speed || 1) + 5000);
      if (this.ws?.readyState === WebSocket.OPEN && this.lastHeartbeatAt > 0
          && Date.now() - this.lastHeartbeatAt > 12000) {
        this.websocketStatus = 'STALE';
      }
      const identityBehind = restId != null && this.lastStateId != null && !sameIdentity;
      const oldFrame = this.lastStateAt > 0 && Date.now() - this.lastStateAt > grace;
      this.stateStatus = (this.websocketStatus === 'STALE' || identityBehind || (state.simulation?.running === true && oldFrame))
        ? 'STALE' : sameIdentity ? 'LIVE' : oldFrame ? 'STALE' : 'WAITING';
    } catch (_) {
      this.restStatus = 'UNAVAILABLE';
      this.stateStatus = 'STALE';
    } finally {
      clearTimeout(timeout);
      this.healthPending = false;
      this.publishHealth();
    }
  }

  async post(endpoint, data) {
    const id = ++this.commandId;
    const report = message => { if (id === this.commandId) this.onCommand(message); };
    report('COMMAND SENT');
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data),
      });
      const result = await response.json();
      if (!response.ok) { report(`COMMAND REJECTED (${response.status})`); return result; }
      report('ACCEPTED · application shown by backend state');
      return result;
    } catch (err) {
      report('COMMAND FAILED · check connection');
      console.error(`API Error on ${endpoint}:`, err);
    }
  }

  async setGate(reservoirId, value) { return this.post(`/gate/${reservoirId}`, {value}); }
  async play() { return this.post('/simulation/play', {}); }
  async pause() { return this.post('/simulation/pause', {}); }
  async step() { return this.post('/simulation/step', {}); }
  async reset() { return this.post('/simulation/reset', {}); }
  async setSpeed(speed) { return this.post('/simulation/speed', {speed}); }
  async setStorm(value) { return this.post('/storm', {value: value / 100.0}); }
  async setMode(mode, source) { return this.post('/controller/mode', source === undefined ? {mode} : {mode, source}); }
}
