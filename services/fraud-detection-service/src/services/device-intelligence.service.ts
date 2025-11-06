import { logger } from '@shared/utils/logger';
import { DeviceFingerprint, DeviceRiskScore } from '@shared/types/fraud.types';
import { createHash } from 'crypto';

export class DeviceIntelligenceService {
  private readonly suspiciousDeviceThreshold = 0.7;
  private deviceCache = new Map<string, DeviceFingerprint>();

  async analyzeDevice(fingerprint: DeviceFingerprint): Promise<DeviceRiskScore> {
    try {
      const deviceId = this.generateDeviceId(fingerprint);
      const knownDevice = this.deviceCache.get(deviceId);

      const riskFactors = {
        isNewDevice: !knownDevice,
        hasVPN: this.detectVPN(fingerprint),
        hasProxy: this.detectProxy(fingerprint),
        hasEmulator: this.detectEmulator(fingerprint),
        hasTamperedBrowser: this.detectBrowserTampering(fingerprint),
        suspiciousUserAgent: this.analyzUserAgent(fingerprint.userAgent),
        inconsistentTimezone: this.checkTimezoneConsistency(fingerprint),
        multipleAccounts: await this.checkMultipleAccounts(deviceId)
      };

      const riskScore = this.calculateDeviceRiskScore(riskFactors);

      if (!knownDevice) {
        this.deviceCache.set(deviceId, fingerprint);
      }

      return {
        deviceId,
        riskScore,
        riskFactors,
        isSuspicious: riskScore > this.suspiciousDeviceThreshold,
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Device analysis failed', { error });
      throw error;
    }
  }

  private generateDeviceId(fingerprint: DeviceFingerprint): string {
    const components = [
      fingerprint.userAgent,
      fingerprint.screenResolution,
      fingerprint.timezone,
      fingerprint.language,
      fingerprint.platform,
      fingerprint.plugins?.join(','),
      fingerprint.canvas,
      fingerprint.webgl
    ].join('|');

    return createHash('sha256').update(components).digest('hex');
  }

  private detectVPN(fingerprint: DeviceFingerprint): boolean {
    // Check for common VPN indicators
    const vpnIndicators = [
      fingerprint.ip?.includes('10.'),
      fingerprint.ip?.includes('172.'),
      fingerprint.timezone !== fingerprint.ipTimezone,
      fingerprint.dnsLeaks?.length > 0
    ];

    return vpnIndicators.filter(Boolean).length >= 2;
  }

  private detectProxy(fingerprint: DeviceFingerprint): boolean {
    const proxyHeaders = [
      'X-Forwarded-For',
      'X-Proxy-ID',
      'Via',
      'Forwarded'
    ];

    return proxyHeaders.some(header => 
      fingerprint.headers?.[header] !== undefined
    );
  }

  private detectEmulator(fingerprint: DeviceFingerprint): boolean {
    const emulatorIndicators = [
      fingerprint.userAgent?.includes('Emulator'),
      fingerprint.platform === 'Android' && fingerprint.hardwareConcurrency === 2,
      fingerprint.deviceMemory === 2,
      fingerprint.touchSupport === false && fingerprint.platform?.includes('Mobile')
    ];

    return emulatorIndicators.filter(Boolean).length >= 2;
  }

  private detectBrowserTampering(fingerprint: DeviceFingerprint): boolean {
    // Check for automation tools
    const automationIndicators = [
      fingerprint.webdriver === true,
      fingerprint.userAgent?.includes('HeadlessChrome'),
      fingerprint.plugins?.length === 0,
      fingerprint.languages?.length === 0
    ];

    return automationIndicators.filter(Boolean).length >= 2;
  }

  private analyzUserAgent(userAgent: string): boolean {
    const suspiciousPatterns = [
      /bot/i,
      /crawler/i,
      /spider/i,
      /scraper/i,
      /curl/i,
      /wget/i,
      /python/i
    ];

    return suspiciousPatterns.some(pattern => pattern.test(userAgent));
  }

  private checkTimezoneConsistency(fingerprint: DeviceFingerprint): boolean {
    if (!fingerprint.timezone || !fingerprint.ipTimezone) return false;

    const timezoneOffset = new Date().getTimezoneOffset();
    const expectedOffset = this.getTimezoneOffset(fingerprint.ipTimezone);

    return Math.abs(timezoneOffset - expectedOffset) > 60; // More than 1 hour difference
  }

  private getTimezoneOffset(timezone: string): number {
    // Simplified timezone offset calculation
    const offsets: Record<string, number> = {
      'America/New_York': 300,
      'America/Los_Angeles': 480,
      'Europe/London': 0,
      'Asia/Tokyo': -540
    };

    return offsets[timezone] || 0;
  }

  private async checkMultipleAccounts(deviceId: string): Promise<boolean> {
    // Check if device is associated with multiple user accounts
    // This would query a database in production
    return false;
  }

  private calculateDeviceRiskScore(factors: Record<string, boolean>): number {
    const weights: Record<string, number> = {
      isNewDevice: 0.1,
      hasVPN: 0.15,
      hasProxy: 0.15,
      hasEmulator: 0.2,
      hasTamperedBrowser: 0.15,
      suspiciousUserAgent: 0.1,
      inconsistentTimezone: 0.1,
      multipleAccounts: 0.05
    };

    return Object.entries(factors).reduce((score, [key, value]) => {
      return score + (value ? weights[key] || 0 : 0);
    }, 0);
  }

  async trackDeviceHistory(deviceId: string, userId: string): Promise<void> {
    logger.info('Tracking device history', { deviceId, userId });
    // Implementation for tracking device usage history
  }

  async getDeviceReputation(deviceId: string): Promise<number> {
    // Query device reputation from threat intelligence feeds
    return 0.5;
  }
}
