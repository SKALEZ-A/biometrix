/**
 * React Native Biometric Fraud Prevention Module
 * Mobile-specific behavioral biometrics and device security
 * Compatible with iOS and Android
 * No external dependencies beyond React Native core
 */

import { NativeModules, NativeEventEmitter, DeviceEventEmitter, Platform, Dimensions } from 'react-native';
import { useEffect, useState, useRef } from 'react';

// Core module constants
const MODULE_NAME = 'BiometricFraudModule';
const SUPPORTED_SENSORS = ['accelerometer', 'gyroscope', 'magnetometer', 'barometer', 'proximity'];
const FRAUD_INDICATORS = {
  SHAKE: { threshold: 2.0, duration: 1000 },
  EMULATOR: { batteryTemp: 0, charging: false },
  ROOTED: { packageListAnomaly: true, debuggable: true },
  ANOMALOUS_TOUCH: { pressureVariance: 0.1, multiTouchSuspicious: true }
};

// Main Biometric Module Class
export class BiometricModule {
  constructor(config = {}) {
    this.config = {
      apiEndpoint: 'http://localhost:3001/api/v1/biometric',
      sessionId: this.generateUUID(),
      userId: config.userId || 'mobile_' + this.generateShortId(),
      enabledSensors: SUPPORTED_SENSORS,
      samplingRate: 100, // ms
      privacyLevel: 'hashed',
      maxBufferSize: 500,
      autoSync: true,
      ...config
    };

    this.isActive = false;
    this.eventBuffer = [];
    this.sensorSubscriptions = {};
    this.syncTimer = null;
    this.deviceInfo = null;
    this.fingerprint = null;
    this.logger = config.debug ? console.log.bind(console, '[BiometricModule]') : () => {};

    this.initializeDeviceInfo();
    this.generateFingerprint();
  }

  // Utility: Generate UUID for sessions
  generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  generateShortId() {
    return Math.random().toString(36).substr(2, 9);
  }

  // Initialize device information
  async initializeDeviceInfo() {
    try {
      // Basic device info
      this.deviceInfo = {
        platform: Platform.OS,
        version: Platform.Version,
        model: await this.getDeviceModel(),
        brand: await this.getDeviceBrand(),
        manufacturer: await this.getDeviceManufacturer(),
        systemName: Platform.constants.SystemName || 'Unknown',
        screen: Dimensions.get('window'),
        uniqueId: await this.getUniqueId(),
        timestamp: Date.now()
      };

      // Security checks
      await this.checkEmulator();
      await this.checkRooted();

      this.logger('Device info initialized:', this.deviceInfo);
      this.bufferEvent('device_info', this.deviceInfo);
    } catch (error) {
      this.logger('Error initializing device info:', error);
    }
  }

  // Generate device fingerprint
  async generateFingerprint() {
    try {
      // Canvas-like fingerprint using device sensors
      const canvasData = await this.simulateCanvasFingerprint();
      const sensorBaseline = await this.getSensorBaseline();
      
      this.fingerprint = {
        hash: this.hashData(JSON.stringify({ ...this.deviceInfo, canvas: canvasData, sensors: sensorBaseline })),
        components: {
          hardware: `${this.deviceInfo.manufacturer}-${this.deviceInfo.model}`,
          os: `${this.deviceInfo.platform}-${this.deviceInfo.version}`,
          screen: `${this.deviceInfo.screen.width}x${this.deviceInfo.screen.height}`,
          sensors: Object.keys(sensorBaseline).join('-'),
          security: `${this.isEmulator ? 'emulated' : 'physical'}-${this.isRooted ? 'rooted' : 'secure'}`
        },
        generatedAt: Date.now(),
        version: '1.0'
      };

      this.logger('Device fingerprint generated:', this.fingerprint.hash);
      this.bufferEvent('device_fingerprint', this.fingerprint);
    } catch (error) {
      this.logger('Error generating fingerprint:', error);
    }
  }

  // Sensor management
  async startSensors() {
    if (this.isActive) return;

    this.isActive = true;
    this.startTime = Date.now();

    for (const sensor of this.config.enabledSensors) {
      try {
        const subscription = await this.subscribeSensor(sensor);
        if (subscription) {
          this.sensorSubscriptions[sensor] = subscription;
          this.logger(`Started ${sensor} sensor`);
        }
      } catch (error) {
        this.logger(`Failed to start ${sensor}:`, error);
      }
    }

    // Start auto-sync
    if (this.config.autoSync) {
      this.startAutoSync();
    }

    // Periodic device checks
    this.periodicChecks = setInterval(() => {
      this.checkDeviceIntegrity();
      this.bufferEvent('device_check', { timestamp: Date.now(), integrity: this.getIntegrityScore() });
    }, 30000); // Every 30 seconds

    this.logger('Sensor monitoring started');
  }

  async stopSensors() {
    if (!this.isActive) return;

    this.isActive = false;

    // Unsubscribe sensors
    Object.keys(this.sensorSubscriptions).forEach(sensor => {
      this.sensorSubscriptions[sensor]?.remove?.();
      delete this.sensorSubscriptions[sensor];
    });

    // Stop auto-sync and checks
    this.stopAutoSync();
    if (this.periodicChecks) {
      clearInterval(this.periodicChecks);
      this.periodicChecks = null;
    }

    // Force sync on stop
    await this.syncBuffer(true);

    this.logger('Sensor monitoring stopped');
  }

  async subscribeSensor(sensorType) {
    // Simulate native sensor subscriptions using DeviceEventEmitter
    // In real implementation, use react-native-sensors or native modules
    
    const handler = (data) => {
      const eventData = {
        type: sensorType,
        timestamp: Date.now(),
        values: data,
        accuracy: 'high', // Simulate
        sessionId: this.config.sessionId,
        userId: this.config.userId
      };

      // Add fraud detection logic
      this.analyzeSensorForFraud(sensorType, data);

      this.bufferEvent('sensor_data', eventData);
    };

    // Simulate different sensor frequencies
    const interval = sensorType === 'accelerometer' ? 100 : sensorType === 'gyroscope' ? 200 : 500;
    
    const subscription = DeviceEventEmitter.addListener(`sensor:${sensorType}`, handler);
    
    // Simulate sensor data generation
    const simulateData = setInterval(() => {
      if (!this.isActive) {
        clearInterval(simulateData);
        return;
      }
      
      let simulatedData;
      switch (sensorType) {
        case 'accelerometer':
          simulatedData = {
            x: (Math.random() - 0.5) * 2 * 9.8, // g-force
            y: (Math.random() - 0.5) * 2 * 9.8,
            z: (Math.random() - 0.5) * 2 * 9.8 + 9.8 // gravity
          };
          break;
        case 'gyroscope':
          simulatedData = {
            x: (Math.random() - 0.5) * 2,
            y: (Math.random() - 0.5) * 2,
            z: (Math.random() - 0.5) * 2
          };
          break;
        case 'magnetometer':
          simulatedData = {
            x: (Math.random() - 0.5) * 100,
            y: (Math.random() - 0.5) * 100,
            z: (Math.random() - 0.5) * 100
          };
          break;
        default:
          simulatedData = { value: Math.random() };
      }

      DeviceEventEmitter.emit(`sensor:${sensorType}`, simulatedData);
    }, interval);

    return {
      remove: () => {
        DeviceEventEmitter.removeSubscription(subscription);
        clearInterval(simulateData);
      }
    };
  }

  // Event buffering with privacy
  bufferEvent(type, data) {
    if (!this.isActive) return;

    const privacyData = this.config.privacyLevel === 'hashed' ? 
      this.hashData({ ...data, userId: this.config.userId }) : data;

    const event = {
      type,
      sessionId: this.config.sessionId,
      userId: this.config.userId,
      deviceId: this.fingerprint?.hash,
      timestamp: Date.now(),
      data: privacyData,
      integrity: this.getIntegrityScore()
    };

    this.eventBuffer.push(event);

    // Buffer management
    if (this.eventBuffer.length > this.config.maxBufferSize) {
      this.eventBuffer.shift();
      this.logger('Buffer overflow, dropped oldest event');
    }

    this.logger(`Buffered ${type} event`);
  }

  // Fraud analysis on sensor data
  analyzeSensorForFraud(sensorType, data) {
    let fraudScore = 0;
    let indicators = [];

    switch (sensorType) {
      case 'accelerometer':
        // Shake detection
        const accelMagnitude = Math.sqrt(data.x**2 + data.y**2 + data.z**2);
        if (accelMagnitude > FRAUD_INDICATORS.SHAKE.threshold) {
          fraudScore += 30;
          indicators.push('shake_detected');
        }

        // Free fall detection
        if (accelMagnitude < 0.5) {
          fraudScore += 20;
          indicators.push('free_fall');
        }
        break;

      case 'gyroscope':
        // Rapid rotation (emulator artifact)
        const gyroMagnitude = Math.sqrt(data.x**2 + data.y**2 + data.z**2);
        if (gyroMagnitude > 10) {
          fraudScore += 25;
          indicators.push('rapid_rotation');
        }
        break;

      case 'magnetometer':
        // Magnetic field anomalies
        const magField = Math.sqrt(data.x**2 + data.y**2 + data.z**2);
        if (magField < 20 || magField > 60) { // Earth normal ~25-65 μT
          fraudScore += 15;
          indicators.push('magnetic_anomaly');
        }
        break;
    }

    if (fraudScore > 0) {
      this.bufferEvent('fraud_indicator', {
        sensor: sensorType,
        score: fraudScore,
        indicators,
        timestamp: Date.now()
      });
    }
  }

  // Auto-sync mechanism
  startAutoSync() {
    this.syncTimer = setInterval(() => {
      if (this.eventBuffer.length > 0) {
        this.syncBuffer();
      }
    }, 5000); // Every 5 seconds
  }

  stopAutoSync() {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  async syncBuffer(force = false) {
    if (this.eventBuffer.length === 0 && !force) return;

    const batch = [...this.eventBuffer];
    this.eventBuffer = [];

    try {
      const response = await fetch(this.config.apiEndpoint + '/mobile-events', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiToken || 'mobile-token'}`,
          'X-Platform': Platform.OS,
          'X-Session-ID': this.config.sessionId
        },
        body: JSON.stringify({
          userId: this.config.userId,
          sessionId: this.config.sessionId,
          platform: Platform.OS,
          deviceInfo: this.deviceInfo,
          events: batch
        })
      });

      if (response.ok) {
        const result = await response.json();
        this.logger('Mobile events synced:', result);
        return result;
      } else {
        this.logger('Sync failed:', response.status);
        if (!force) this.eventBuffer.unshift(...batch); // Retry
      }
    } catch (error) {
      this.logger('Network error during sync:', error);
      if (!force) this.eventBuffer.unshift(...batch);
    }
  }

  // Security checks
  async checkEmulator() {
    // Simulate emulator detection
    this.isEmulator = Math.random() > 0.95; // 5% chance for demo
    this.logger(`Emulator check: ${this.isEmulator ? 'DETECTED' : 'clean'}`);
    
    if (this.isEmulator) {
      this.bufferEvent('security_alert', { type: 'emulator_detected', severity: 'high' });
    }
  }

  async checkRooted() {
    // Simulate root detection
    this.isRooted = Math.random() > 0.92; // 8% chance for demo
    this.logger(`Root check: ${this.isRooted ? 'DETECTED' : 'clean'}`);
    
    if (this.isRooted) {
      this.bufferEvent('security_alert', { type: 'device_rooted', severity: 'high' });
    }
  }

  async checkDeviceIntegrity() {
    const checks = await Promise.all([
      this.checkEmulator(),
      this.checkRooted(),
      this.checkJailbreak(),
      this.checkAppIntegrity()
    ]);

    const anomalies = checks.filter(check => check.anomaly).length;
    if (anomalies > 0) {
      this.bufferEvent('integrity_violation', { 
        anomalies, 
        checks,
        timestamp: Date.now() 
      });
    }
  }

  getIntegrityScore() {
    let score = 100;
    
    if (this.isEmulator) score -= 40;
    if (this.isRooted) score -= 50;
    if (this.isJailbroken) score -= 45;
    
    return Math.max(0, score);
  }

  // Privacy: Hash sensitive data
  hashData(data) {
    // Simple SHA-like hash for demo (use crypto in production)
    let hash = 0;
    const str = JSON.stringify(data);
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return 'hash_' + Math.abs(hash).toString(36).substring(0, 32);
  }

  // Simulated native methods (replace with actual NativeModules)
  async getDeviceModel() {
    return Platform.OS === 'ios' ? 'iPhone14,5' : 'SM-G998B'; // Demo values
  }

  async getDeviceBrand() {
    return Platform.OS === 'ios' ? 'Apple' : 'Samsung';
  }

  async getDeviceManufacturer() {
    return Platform.OS === 'ios' ? 'Apple' : 'samsung';
  }

  async getUniqueId() {
    return 'mobile_' + this.generateUUID().substring(0, 8);
  }

  async simulateCanvasFingerprint() {
    // Simulate canvas fingerprinting on mobile
    return {
      fontList: ['Arial', 'Helvetica', 'Times New Roman'], // Simulated
      rendering: Platform.OS,
      dpi: Platform.OS === 'ios' ? 326 : 420, // Demo
      timestamp: Date.now()
    };
  }

  async getSensorBaseline() {
    return {
      accelerometer: { x: 0, y: 0, z: 9.8 },
      gyroscope: { x: 0, y: 0, z: 0 },
      magnetometer: { x: 20, y: 30, z: 50 } // Earth field
    };
  }

  async checkJailbreak() {
    this.isJailbroken = Platform.OS === 'ios' && Math.random() > 0.98;
    return { anomaly: this.isJailbroken, type: 'jailbreak' };
  }

  async checkAppIntegrity() {
    // Simulate signature verification
    const isTampered = Math.random() > 0.99;
    return { anomaly: isTampered, type: 'app_tampered' };
  }

  // React Hook for easy integration
  useBiometricModule(config = {}) {
    const moduleRef = useRef(null);
    const [status, setStatus] = useState('idle');
    const [stats, setStats] = useState({});

    useEffect(() => {
      moduleRef.current = new BiometricModule({ ...config, debug: __DEV__ });
      
      const startMonitoring = async () => {
        await moduleRef.current.startSensors();
        setStatus('active');
        
        // Periodic stats update
        const interval = setInterval(() => {
          setStats({
            eventCount: moduleRef.current.eventBuffer.length,
            sessionDuration: Date.now() - moduleRef.current.startTime,
            integrity: moduleRef.current.getIntegrityScore(),
            fingerprint: moduleRef.current.fingerprint?.hash
          });
        }, 5000);

        return () => clearInterval(interval);
      };

      startMonitoring();

      return () => {
        if (moduleRef.current) {
          moduleRef.current.stopSensors();
          setStatus('stopped');
        }
      };
    }, []);

    return {
      status,
      stats,
      start: () => moduleRef.current?.startSensors(),
      stop: () => moduleRef.current?.stopSensors(),
      sync: () => moduleRef.current?.syncBuffer(true),
      getFingerprint: () => moduleRef.current?.fingerprint,
      getDeviceInfo: () => moduleRef.current?.deviceInfo
    };
  }

  // Cleanup
  destroy() {
    this.stopSensors();
    this.eventBuffer = [];
    this.sensorSubscriptions = {};
    this.deviceInfo = null;
    this.fingerprint = null;
    this.logger('BiometricModule destroyed');
  }
}

// Export for CommonJS/ES modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = BiometricModule;
}

export default BiometricModule;

// Demo usage in React Native component
/*
import React from 'react';
import { View, Text, Button } from 'react-native';
import BiometricModule, { useBiometricModule } from './BiometricModule';

const FraudShield = () => {
  const { status, stats, start, stop, sync } = useBiometricModule({
    userId: 'user_123',
    apiToken: 'your-token'
  });

  return (
    <View style={{ padding: 20 }}>
      <Text>Status: {status}</Text>
      <Text>Events: {stats.eventCount}</Text>
      <Text>Integrity: {stats.integrity}%</Text>
      <Button title="Start Monitoring" onPress={start} disabled={status === 'active'} />
      <Button title="Stop & Sync" onPress={() => { stop(); sync(); }} disabled={status !== 'active'} />
    </View>
  );
};

export default FraudShield;
*/
