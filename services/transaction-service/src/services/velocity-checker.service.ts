import { VelocityRule, VelocityViolation, TimeWindow } from '@shared/types/fraud.types';
import { Logger } from '@shared/utils/logger';
import { RedisClient } from '@shared/cache/redis';

export class VelocityCheckerService {
  private readonly logger = new Logger('VelocityCheckerService');
  private readonly redis: RedisClient;
  private readonly rules: VelocityRule[] = [];

  constructor() {
    this.redis = new RedisClient();
    this.initializeDefaultRules();
  }

  private initializeDefaultRules(): void {
    this.rules.push(
      {
        id: 'txn_count_1h',
        name: 'Transaction Count - 1 Hour',
        window: { value: 1, unit: 'hour' },
        metric: 'count',
        threshold: 10,
        severity: 'medium'
      },
      {
        id: 'txn_amount_1h',
        name: 'Transaction Amount - 1 Hour',
        window: { value: 1, unit: 'hour' },
        metric: 'sum',
        threshold: 10000,
        severity: 'high'
      },
      {
        id: 'txn_count_24h',
        name: 'Transaction Count - 24 Hours',
        window: { value: 24, unit: 'hour' },
        metric: 'count',
        threshold: 50,
        severity: 'low'
      },
      {
        id: 'unique_recipients_1h',
        name: 'Unique Recipients - 1 Hour',
        window: { value: 1, unit: 'hour' },
        metric: 'unique',
        threshold: 15,
        severity: 'high'
      },
      {
        id: 'failed_attempts_15m',
        name: 'Failed Attempts - 15 Minutes',
        window: { value: 15, unit: 'minute' },
        metric: 'count',
        threshold: 5,
        severity: 'critical'
      }
    );
  }

  async checkVelocity(userId: string, transaction: any): Promise<VelocityViolation[]> {
    const violations: VelocityViolation[] = [];

    for (const rule of this.rules) {
      const violation = await this.checkRule(userId, transaction, rule);
      if (violation) {
        violations.push(violation);
      }
    }

    return violations;
  }

  private async checkRule(
    userId: string,
    transaction: any,
    rule: VelocityRule
  ): Promise<VelocityViolation | null> {
    const windowMs = this.convertToMilliseconds(rule.window);
    const now = Date.now();
    const windowStart = now - windowMs;

    const key = `velocity:${userId}:${rule.id}`;
    const transactions = await this.getTransactionsInWindow(key, windowStart, now);

    transactions.push(transaction);

    let currentValue: number;
    switch (rule.metric) {
      case 'count':
        currentValue = transactions.length;
        break;
      case 'sum':
        currentValue = transactions.reduce((sum, tx) => sum + (tx.amount || 0), 0);
        break;
      case 'unique':
        currentValue = new Set(transactions.map(tx => tx.recipientId)).size;
        break;
      case 'average':
        currentValue = transactions.reduce((sum, tx) => sum + (tx.amount || 0), 0) / transactions.length;
        break;
      default:
        currentValue = 0;
    }

    if (currentValue > rule.threshold) {
      return {
        ruleId: rule.id,
        ruleName: rule.name,
        threshold: rule.threshold,
        currentValue,
        severity: rule.severity,
        window: rule.window,
        timestamp: now
      };
    }

    await this.storeTransaction(key, transaction, windowMs);
    return null;
  }

  private convertToMilliseconds(window: TimeWindow): number {
    const multipliers = {
      second: 1000,
      minute: 60 * 1000,
      hour: 60 * 60 * 1000,
      day: 24 * 60 * 60 * 1000
    };

    return window.value * multipliers[window.unit];
  }

  private async getTransactionsInWindow(
    key: string,
    windowStart: number,
    windowEnd: number
  ): Promise<any[]> {
    try {
      const data = await this.redis.zrangebyscore(key, windowStart, windowEnd);
      return data.map(item => JSON.parse(item));
    } catch (error) {
      this.logger.error('Failed to get transactions from Redis', error);
      return [];
    }
  }

  private async storeTransaction(key: string, transaction: any, ttl: number): Promise<void> {
    try {
      const score = Date.now();
      await this.redis.zadd(key, score, JSON.stringify(transaction));
      await this.redis.expire(key, Math.ceil(ttl / 1000));
    } catch (error) {
      this.logger.error('Failed to store transaction in Redis', error);
    }
  }

  async checkCrossBorderVelocity(userId: string, country: string): Promise<boolean> {
    const key = `velocity:crossborder:${userId}`;
    const windowMs = 24 * 60 * 60 * 1000;
    const now = Date.now();
    const windowStart = now - windowMs;

    const countries = await this.getCountriesInWindow(key, windowStart, now);
    countries.push(country);

    const uniqueCountries = new Set(countries);

    if (uniqueCountries.size > 5) {
      this.logger.warn(`User ${userId} transacted in ${uniqueCountries.size} countries in 24h`);
      return true;
    }

    await this.storeCountry(key, country, windowMs);
    return false;
  }

  private async getCountriesInWindow(
    key: string,
    windowStart: number,
    windowEnd: number
  ): Promise<string[]> {
    try {
      return await this.redis.zrangebyscore(key, windowStart, windowEnd);
    } catch (error) {
      this.logger.error('Failed to get countries from Redis', error);
      return [];
    }
  }

  private async storeCountry(key: string, country: string, ttl: number): Promise<void> {
    try {
      const score = Date.now();
      await this.redis.zadd(key, score, country);
      await this.redis.expire(key, Math.ceil(ttl / 1000));
    } catch (error) {
      this.logger.error('Failed to store country in Redis', error);
    }
  }

  async checkDeviceVelocity(userId: string, deviceId: string): Promise<boolean> {
    const key = `velocity:device:${userId}`;
    const windowMs = 60 * 60 * 1000;
    const now = Date.now();
    const windowStart = now - windowMs;

    const devices = await this.getDevicesInWindow(key, windowStart, now);
    devices.push(deviceId);

    const uniqueDevices = new Set(devices);

    if (uniqueDevices.size > 3) {
      this.logger.warn(`User ${userId} used ${uniqueDevices.size} devices in 1h`);
      return true;
    }

    await this.storeDevice(key, deviceId, windowMs);
    return false;
  }

  private async getDevicesInWindow(
    key: string,
    windowStart: number,
    windowEnd: number
  ): Promise<string[]> {
    try {
      return await this.redis.zrangebyscore(key, windowStart, windowEnd);
    } catch (error) {
      this.logger.error('Failed to get devices from Redis', error);
      return [];
    }
  }

  private async storeDevice(key: string, deviceId: string, ttl: number): Promise<void> {
    try {
      const score = Date.now();
      await this.redis.zadd(key, score, deviceId);
      await this.redis.expire(key, Math.ceil(ttl / 1000));
    } catch (error) {
      this.logger.error('Failed to store device in Redis', error);
    }
  }

  async checkIPVelocity(userId: string, ipAddress: string): Promise<boolean> {
    const key = `velocity:ip:${userId}`;
    const windowMs = 30 * 60 * 1000;
    const now = Date.now();
    const windowStart = now - windowMs;

    const ips = await this.getIPsInWindow(key, windowStart, now);
    ips.push(ipAddress);

    const uniqueIPs = new Set(ips);

    if (uniqueIPs.size > 5) {
      this.logger.warn(`User ${userId} used ${uniqueIPs.size} IP addresses in 30m`);
      return true;
    }

    await this.storeIP(key, ipAddress, windowMs);
    return false;
  }

  private async getIPsInWindow(
    key: string,
    windowStart: number,
    windowEnd: number
  ): Promise<string[]> {
    try {
      return await this.redis.zrangebyscore(key, windowStart, windowEnd);
    } catch (error) {
      this.logger.error('Failed to get IPs from Redis', error);
      return [];
    }
  }

  private async storeIP(key: string, ipAddress: string, ttl: number): Promise<void> {
    try {
      const score = Date.now();
      await this.redis.zadd(key, score, ipAddress);
      await this.redis.expire(key, Math.ceil(ttl / 1000));
    } catch (error) {
      this.logger.error('Failed to store IP in Redis', error);
    }
  }

  async calculateVelocityScore(userId: string, transaction: any): Promise<number> {
    let score = 0;

    const violations = await this.checkVelocity(userId, transaction);
    
    for (const violation of violations) {
      const severityScores = {
        low: 0.2,
        medium: 0.4,
        high: 0.6,
        critical: 0.9
      };
      
      const baseScore = severityScores[violation.severity];
      const exceedanceRatio = violation.currentValue / violation.threshold;
      score += baseScore * Math.min(exceedanceRatio, 2);
    }

    const crossBorderViolation = await this.checkCrossBorderVelocity(userId, transaction.country);
    if (crossBorderViolation) score += 0.3;

    const deviceViolation = await this.checkDeviceVelocity(userId, transaction.deviceId);
    if (deviceViolation) score += 0.25;

    const ipViolation = await this.checkIPVelocity(userId, transaction.ipAddress);
    if (ipViolation) score += 0.2;

    return Math.min(score, 1.0);
  }

  async addCustomRule(rule: VelocityRule): Promise<void> {
    this.rules.push(rule);
    this.logger.info(`Added custom velocity rule: ${rule.name}`);
  }

  async removeRule(ruleId: string): Promise<boolean> {
    const index = this.rules.findIndex(r => r.id === ruleId);
    if (index !== -1) {
      this.rules.splice(index, 1);
      this.logger.info(`Removed velocity rule: ${ruleId}`);
      return true;
    }
    return false;
  }

  getRules(): VelocityRule[] {
    return [...this.rules];
  }

  async clearUserVelocityData(userId: string): Promise<void> {
    const patterns = [
      `velocity:${userId}:*`,
      `velocity:crossborder:${userId}`,
      `velocity:device:${userId}`,
      `velocity:ip:${userId}`
    ];

    for (const pattern of patterns) {
      try {
        const keys = await this.redis.keys(pattern);
        for (const key of keys) {
          await this.redis.del(key);
        }
      } catch (error) {
        this.logger.error(`Failed to clear velocity data for pattern ${pattern}`, error);
      }
    }

    this.logger.info(`Cleared velocity data for user ${userId}`);
  }
}
