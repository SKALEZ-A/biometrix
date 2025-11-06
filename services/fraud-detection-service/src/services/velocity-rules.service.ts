import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface VelocityRule {
  id: string;
  name: string;
  dimension: string;
  threshold: number;
  windowSeconds: number;
  action: 'block' | 'review' | 'alert';
  severity: 'low' | 'medium' | 'high' | 'critical';
}

interface VelocityCheck {
  dimension: string;
  key: string;
  value: number;
  timestamp: number;
}

export class VelocityRulesService {
  private redis: RedisClient;
  private rules: Map<string, VelocityRule>;

  constructor() {
    this.redis = new RedisClient();
    this.rules = new Map();
    this.initializeDefaultRules();
  }

  private initializeDefaultRules(): void {
    const defaultRules: VelocityRule[] = [
      {
        id: 'txn_per_card_1h',
        name: 'Transactions per card per hour',
        dimension: 'card',
        threshold: 10,
        windowSeconds: 3600,
        action: 'review',
        severity: 'medium'
      },
      {
        id: 'txn_per_ip_5m',
        name: 'Transactions per IP per 5 minutes',
        dimension: 'ip',
        threshold: 5,
        windowSeconds: 300,
        action: 'block',
        severity: 'high'
      },
      {
        id: 'txn_per_user_1d',
        name: 'Transactions per user per day',
        dimension: 'user',
        threshold: 50,
        windowSeconds: 86400,
        action: 'alert',
        severity: 'low'
      },
      {
        id: 'failed_txn_per_card_15m',
        name: 'Failed transactions per card per 15 minutes',
        dimension: 'card:failed',
        threshold: 3,
        windowSeconds: 900,
        action: 'block',
        severity: 'critical'
      },
      {
        id: 'amount_per_card_1h',
        name: 'Total amount per card per hour',
        dimension: 'card:amount',
        threshold: 10000,
        windowSeconds: 3600,
        action: 'review',
        severity: 'high'
      }
    ];

    defaultRules.forEach(rule => this.rules.set(rule.id, rule));
  }

  public async checkVelocity(check: VelocityCheck): Promise<{
    violated: boolean;
    rules: VelocityRule[];
    currentCount: number;
  }> {
    const violatedRules: VelocityRule[] = [];
    const relevantRules = Array.from(this.rules.values())
      .filter(rule => rule.dimension === check.dimension);

    let maxCount = 0;

    for (const rule of relevantRules) {
      const key = `velocity:${rule.dimension}:${check.key}`;
      const count = await this.incrementVelocity(key, check.value, rule.windowSeconds);
      
      maxCount = Math.max(maxCount, count);

      if (count > rule.threshold) {
        violatedRules.push(rule);
      }
    }

    return {
      violated: violatedRules.length > 0,
      rules: violatedRules,
      currentCount: maxCount
    };
  }

  private async incrementVelocity(
    key: string,
    value: number,
    windowSeconds: number
  ): Promise<number> {
    const now = Date.now();
    const windowStart = now - (windowSeconds * 1000);

    await this.redis.zadd(key, now, `${now}:${value}`);
    await this.redis.zremrangebyscore(key, 0, windowStart);
    await this.redis.expire(key, windowSeconds);

    const members = await this.redis.zrange(key, 0, -1);
    
    return members.reduce((sum, member) => {
      const [, val] = member.split(':');
      return sum + parseFloat(val);
    }, 0);
  }

  public async getVelocityStats(dimension: string, key: string): Promise<{
    counts: Record<string, number>;
    windows: Record<string, number>;
  }> {
    const relevantRules = Array.from(this.rules.values())
      .filter(rule => rule.dimension === dimension);

    const counts: Record<string, number> = {};
    const windows: Record<string, number> = {};

    for (const rule of relevantRules) {
      const redisKey = `velocity:${rule.dimension}:${key}`;
      const members = await this.redis.zrange(redisKey, 0, -1);
      
      const count = members.reduce((sum, member) => {
        const [, val] = member.split(':');
        return sum + parseFloat(val);
      }, 0);

      counts[rule.id] = count;
      windows[rule.id] = rule.windowSeconds;
    }

    return { counts, windows };
  }

  public addRule(rule: VelocityRule): void {
    this.rules.set(rule.id, rule);
  }

  public removeRule(ruleId: string): boolean {
    return this.rules.delete(ruleId);
  }

  public getRule(ruleId: string): VelocityRule | undefined {
    return this.rules.get(ruleId);
  }

  public getAllRules(): VelocityRule[] {
    return Array.from(this.rules.values());
  }

  public async resetVelocity(dimension: string, key: string): Promise<void> {
    const pattern = `velocity:${dimension}:${key}`;
    await this.redis.del(pattern);
  }
}

export const velocityRulesService = new VelocityRulesService();
