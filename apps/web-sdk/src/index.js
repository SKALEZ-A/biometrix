/**
 * Biometric Fraud Prevention Web SDK
 * Enterprise-grade client-side behavioral biometrics capture
 * Version: 2.0.0
 * Author: Fraud Prevention Team
 */

(function(global) {
  'use strict';

  // SDK Configuration
  const DEFAULT_CONFIG = {
    apiEndpoint: 'http://localhost:3001/api/v1/biometric',
    sessionTimeout: 30 * 60 * 1000, // 30 minutes
    samplingRate: 10, // ms between samples
    enabledCollectors: ['keystroke', 'mouse', 'touch', 'device'],
    privacyMode: 'hashed', // 'hashed' | 'pseudonymized' | 'raw'
    maxEventBuffer: 1000,
    autoFlush: true,
    flushInterval: 5000, // 5 seconds
    debug: false
  };

  // Event types and schemas
  const EVENT_TYPES = {
    keystroke: { dwell: true, flight: true, keyCode: true, timestamp: true },
    mouse: { x: true, y: true, velocity: true, acceleration: true, button: true, timestamp: true },
    touch: { x: true, y: true, pressure: true, swipe: true, timestamp: true },
    device: { orientation: true, motion: true, screen: true, timestamp: true },
    page: { visibility: true, focus: true, timestamp: true }
  };

  // Core SDK Class
  class BiometricSDK {
    constructor(config = {}) {
      this.config = { ...DEFAULT_CONFIG, ...config };
      this.isInitialized = false;
      this.isMonitoring = false;
      this.eventBuffer = [];
      this.sessionId = this.generateSessionId();
      this.userId = config.userId || 'anonymous_' + this.generateId();
      this.startTime = Date.now();
      this.collectors = {};
      this.flushInterval = null;
      this.debugLog = this.config.debug ? console.log.bind(console, '[BiometricSDK]') : () => {};

      this.debugLog('SDK initialized with config:', this.config);
    }

    // Generate unique IDs
    generateId(length = 16) {
      return Array.from({ length }, () => Math.floor(Math.random() * 16).toString(16)).join('');
    }

    generateSessionId() {
      return 'session_' + Date.now() + '_' + this.generateId(8);
    }

    // Privacy: Hash sensitive data
    hashData(data) {
      if (this.config.privacyMode === 'hashed') {
        const hash = this.simpleHash(JSON.stringify(data) + this.userId);
        return { hashed: true, hash: hash.substring(0, 32), originalLength: JSON.stringify(data).length };
      }
      return { hashed: false, data };
    }

    simpleHash(str) {
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
      }
      return Math.abs(hash).toString(36);
    }

    // Initialize SDK
    init() {
      if (this.isInitialized) {
        this.debugLog('SDK already initialized');
        return this;
      }

      // Initialize collectors
      this.initCollectors();

      // Setup auto-flush
      if (this.config.autoFlush) {
        this.startAutoFlush();
      }

      // Listen for page visibility changes
      document.addEventListener('visibilitychange', () => this.capturePageEvent());
      window.addEventListener('focus', () => this.capturePageEvent());
      window.addEventListener('blur', () => this.capturePageEvent());

      this.isInitialized = true;
      this.debugLog('SDK fully initialized');
      return this;
    }

    initCollectors() {
      const enabled = this.config.enabledCollectors;

      if (enabled.includes('keystroke')) {
        this.collectors.keystroke = new KeystrokeCollector(this);
        document.addEventListener('keydown', (e) => this.collectors.keystroke.startCapture(e));
        document.addEventListener('keyup', (e) => this.collectors.keystroke.endCapture(e));
      }

      if (enabled.includes('mouse')) {
        this.collectors.mouse = new MouseCollector(this);
        document.addEventListener('mousemove', (e) => this.collectors.mouse.capture(e));
        document.addEventListener('mousedown', (e) => this.collectors.mouse.capture(e));
        document.addEventListener('mouseup', (e) => this.collectors.mouse.capture(e));
      }

      if (enabled.includes('touch') && 'ontouchstart' in window) {
        this.collectors.touch = new TouchCollector(this);
        document.addEventListener('touchstart', (e) => this.collectors.touch.capture(e));
        document.addEventListener('touchmove', (e) => this.collectors.touch.capture(e));
        document.addEventListener('touchend', (e) => this.collectors.touch.capture(e));
      }

      if (enabled.includes('device')) {
        this.collectors.device = new DeviceCollector(this);
        if (window.DeviceOrientationEvent) {
          window.addEventListener('deviceorientation', (e) => this.collectors.device.captureOrientation(e));
        }
        if (window.DeviceMotionEvent) {
          window.addEventListener('devicemotion', (e) => this.collectors.device.captureMotion(e));
        }
        this.collectors.device.captureScreenInfo();
      }
    }

    // Start monitoring
    start() {
      if (this.isMonitoring) {
        this.debugLog('Monitoring already active');
        return this;
      }

      this.isMonitoring = true;
      this.startTime = Date.now();
      this.debugLog('Started monitoring session:', this.sessionId);

      // Trigger initial device fingerprint
      if (this.collectors.device) {
        this.collectors.device.captureFingerprint();
      }

      return this;
    }

    // Stop monitoring
    stop() {
      if (!this.isMonitoring) {
        this.debugLog('Monitoring not active');
        return this;
      }

      this.isMonitoring = false;
      this.flushBuffer(true); // Force flush on stop
      this.debugLog('Stopped monitoring, flushed buffer');
      return this;
    }

    // Capture event and buffer
    captureEvent(type, data) {
      if (!this.isMonitoring) return;

      const event = {
        type,
        sessionId: this.sessionId,
        userId: this.userId,
        timestamp: Date.now(),
        data: this.hashData(data).hashed ? this.hashData(data).hash : this.hashData(data).data
      };

      // Validate schema
      if (!this.validateEventSchema(type, event)) {
        this.debugLog('Invalid event schema, skipping:', event);
        return;
      }

      this.eventBuffer.push(event);

      // Buffer management
      if (this.eventBuffer.length > this.config.maxEventBuffer) {
        this.eventBuffer.shift(); // FIFO
        this.debugLog('Buffer overflow, dropped oldest event');
      }

      this.debugLog('Captured event:', type, event.timestamp);
    }

    validateEventSchema(type, event) {
      const schema = EVENT_TYPES[type];
      if (!schema) return false;

      return Object.keys(schema).every(key => {
        if (key === 'timestamp') return true; // Always present
        return event.data && (typeof event.data[key] !== 'undefined');
      });
    }

    // Flush buffer to backend
    async flushBuffer(force = false) {
      if (this.eventBuffer.length === 0 && !force) return;

      const eventsToSend = [...this.eventBuffer];
      this.eventBuffer = []; // Clear buffer

      try {
        const response = await fetch(`${this.config.apiEndpoint}/events`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiToken || 'demo-token'}`,
            'X-Session-ID': this.sessionId
          },
          body: JSON.stringify({
            userId: this.userId,
            sessionId: this.sessionId,
            events: eventsToSend
          })
        });

        if (response.ok) {
          const result = await response.json();
          this.debugLog('Events flushed successfully:', result);
          return result;
        } else {
          this.debugLog('Flush failed:', response.status, await response.text());
          // Re-add to buffer on failure (non-force)
          if (!force) this.eventBuffer.unshift(...eventsToSend);
        }
      } catch (error) {
        this.debugLog('Network error during flush:', error);
        if (!force) this.eventBuffer.unshift(...eventsToSend);
      }
    }

    // Auto-flush mechanism
    startAutoFlush() {
      this.flushInterval = setInterval(() => {
        if (this.eventBuffer.length > 0) {
          this.flushBuffer();
        }
      }, this.config.flushInterval);
    }

    stopAutoFlush() {
      if (this.flushInterval) {
        clearInterval(this.flushInterval);
        this.flushInterval = null;
      }
    }

    // Get session statistics
    getSessionStats() {
      const duration = Date.now() - this.startTime;
      const eventsByType = this.eventBuffer.reduce((acc, event) => {
        acc[event.type] = (acc[event.type] || 0) + 1;
        return acc;
      }, {});

      return {
        sessionId: this.sessionId,
        userId: this.userId,
        duration: duration,
        eventCount: this.eventBuffer.length,
        eventsByType,
        bufferSize: this.eventBuffer.length,
        startTime: this.startTime,
        isMonitoring: this.isMonitoring,
        config: { ...this.config, apiToken: '[REDACTED]' }
      };
    }

    // Destroy SDK (cleanup)
    destroy() {
      this.stop();
      this.stopAutoFlush();

      // Remove all listeners
      Object.values(this.collectors).forEach(collector => collector.destroy());
      this.collectors = {};

      // Clear event listeners
      document.removeEventListener('keydown', () => {});
      document.removeEventListener('keyup', () => {});
      document.removeEventListener('mousemove', () => {});
      document.removeEventListener('mousedown', () => {});
      document.removeEventListener('mouseup', () => {});
      document.removeEventListener('touchstart', () => {});
      document.removeEventListener('touchmove', () => {});
      document.removeEventListener('touchend', () => {});
      window.removeEventListener('deviceorientation', () => {});
      window.removeEventListener('devicemotion', () => {});
      document.removeEventListener('visibilitychange', () => {});
      window.removeEventListener('focus', () => {});
      window.removeEventListener('blur', () => {});

      this.debugLog('SDK destroyed');
    }

    // Capture page events
    capturePageEvent() {
      this.captureEvent('page', {
        visibility: document.visibilityState,
        focus: document.hasFocus(),
        url: window.location.href,
        referrer: document.referrer
      });
    }
  }

  // Collector Classes
  class KeystrokeCollector {
    constructor(sdk) {
      this.sdk = sdk;
      this.lastKeyTime = 0;
      this.currentKeyCode = null;
    }

    startCapture(event) {
      this.currentKeyCode = event.keyCode || event.which;
      this.lastKeyTime = Date.now();
    }

    endCapture(event) {
      if (!this.currentKeyCode) return;

      const now = Date.now();
      const dwellTime = now - this.lastKeyTime;
      const flightTime = this.lastKeyTime - (this.sdk.lastKeyTime || this.lastKeyTime);

      this.sdk.captureEvent('keystroke', {
        keyCode: this.currentKeyCode,
        dwellTime,
        flightTime,
        char: event.key || String.fromCharCode(this.currentKeyCode),
        location: event.location || 0
      });

      this.sdk.lastKeyTime = now;
      this.currentKeyCode = null;
    }

    destroy() {
      // No specific cleanup needed
    }
  }

  class MouseCollector {
    constructor(sdk) {
      this.sdk = sdk;
      this.lastPosition = { x: 0, y: 0 };
      this.lastTime = Date.now();
      this.buttonState = null;
    }

    capture(event) {
      const now = Date.now();
      const rect = event.target.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const deltaTime = now - this.lastTime;
      const deltaX = x - this.lastPosition.x;
      const deltaY = y - this.lastPosition.y;

      const velocity = Math.sqrt((deltaX * deltaX + deltaY * deltaY) / (deltaTime / 1000));
      const acceleration = deltaTime > 0 ? (velocity - this.velocity || 0) / (deltaTime / 1000) : 0;

      this.sdk.captureEvent('mouse', {
        x, y,
        clientX: event.clientX,
        clientY: event.clientY,
        screenX: event.screenX,
        screenY: event.screenY,
        velocity,
        acceleration: { x: deltaX / deltaTime * 1000, y: deltaY / deltaTime * 1000 },
        button: event.button,
        buttons: event.buttons,
        type: event.type,
        target: event.target.tagName,
        deltaTime
      });

      this.lastPosition = { x, y };
      this.lastTime = now;
      this.velocity = velocity;
      if (event.type === 'mousedown' || event.type === 'mouseup') {
        this.buttonState = event.type === 'mousedown';
      }
    }

    destroy() {
      this.lastPosition = { x: 0, y: 0 };
      this.lastTime = 0;
    }
  }

  class TouchCollector {
    constructor(sdk) {
      this.sdk = sdk;
      this.lastTouch = null;
      this.lastTime = 0;
    }

    capture(event) {
      event.preventDefault(); // Prevent scrolling interference

      const now = Date.now();
      const touch = event.touches[0] || event.changedTouches[0];

      if (!touch) return;

      const rect = event.target.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;

      if (this.lastTouch) {
        const deltaX = x - this.lastTouch.x;
        const deltaY = y - this.lastTouch.y;
        const deltaTime = now - this.lastTime;
        const swipeLength = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const swipeSpeed = swipeLength / (deltaTime / 1000);

        this.sdk.captureEvent('touch', {
          x, y,
          clientX: touch.clientX,
          clientY: touch.clientY,
          pressure: touch.force || touch.pressure || 1.0,
          radiusX: touch.radiusX || 1,
          radiusY: touch.radiusY || 1,
          rotation: touch.rotationAngle || 0,
          swipeLength,
          swipeSpeed,
          touches: event.touches.length,
          type: event.type,
          target: event.target.tagName,
          deltaTime
        });
      } else {
        this.sdk.captureEvent('touch', {
          x, y,
          clientX: touch.clientX,
          clientY: touch.clientY,
          pressure: touch.force || touch.pressure || 1.0,
          touches: event.touches.length,
          type: event.type,
          target: event.target.tagName
        });
      }

      this.lastTouch = { x, y };
      this.lastTime = now;
    }

    destroy() {
      this.lastTouch = null;
      this.lastTime = 0;
    }
  }

  class DeviceCollector {
    constructor(sdk) {
      this.sdk = sdk;
      this.screenInfo = null;
    }

    captureOrientation(event) {
      this.sdk.captureEvent('device', {
        orientation: {
          alpha: event.alpha,
          beta: event.beta,
          gamma: event.gamma
        },
        absolute: event.absolute,
        timestamp: event.timeStamp
      });
    }

    captureMotion(event) {
      this.sdk.captureEvent('device', {
        motion: {
          acceleration: event.acceleration,
          accelerationIncludingGravity: event.accelerationIncludingGravity,
          rotationRate: event.rotationRate
        },
        interval: event.interval,
        timestamp: event.timeStamp
      });
    }

    captureScreenInfo() {
      this.screenInfo = {
        width: screen.width,
        height: screen.height,
        availWidth: screen.availWidth,
        availHeight: screen.availHeight,
        colorDepth: screen.colorDepth,
        pixelDepth: screen.pixelDepth,
        orientation: screen.orientation ? screen.orientation.angle : 0
      };

      this.sdk.captureEvent('device', { screen: this.screenInfo });
    }

    captureFingerprint() {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 200;
      canvas.height = 50;
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('Biometric Fingerprint Test', 2, 2);
      ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
      ctx.fillText('Biometric Fingerprint Test', 4, 4);

      const dataUrl = canvas.toDataURL();
      const hash = this.sdk.simpleHash(dataUrl);

      this.sdk.captureEvent('device', {
        fingerprint: {
          canvasHash: hash,
          userAgent: navigator.userAgent,
          language: navigator.language,
          platform: navigator.platform,
          cookiesEnabled: navigator.cookieEnabled,
          doNotTrack: navigator.doNotTrack,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
          hardwareConcurrency: navigator.hardwareConcurrency,
          deviceMemory: navigator.deviceMemory,
          maxTouchPoints: navigator.maxTouchPoints
        }
      });
    }

    destroy() {
      this.screenInfo = null;
    }
  }

  // Expose SDK globally
  if (typeof global.BiometricSDK !== 'undefined') {
    throw new Error('BiometricSDK already loaded');
  }

  global.BiometricSDK = {
    create: (config) => new BiometricSDK(config),
    version: '2.0.0',
    collectors: EVENT_TYPES
  };

  // Auto-initialize if script loaded with data attributes
  if (document.currentScript) {
    const script = document.currentScript;
    const config = {
      apiEndpoint: script.getAttribute('data-api-endpoint') || DEFAULT_CONFIG.apiEndpoint,
      userId: script.getAttribute('data-user-id'),
      apiToken: script.getAttribute('data-api-token'),
      debug: script.getAttribute('data-debug') === 'true'
    };

    // Remove empty values
    Object.keys(config).forEach(key => {
      if (config[key] === null || config[key] === '') delete config[key];
    });

    const sdk = global.BiometricSDK.create(config);
    sdk.init().start();

    // Expose globally for manual control
    global.biometricSDK = sdk;

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => sdk.stop().destroy());
  }

})(window);

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { BiometricSDK };
}
