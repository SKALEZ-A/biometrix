import { logger } from '@shared/utils/logger';
import { ThreatIndicator, ThreatLevel } from '@shared/types/fraud.types';

export class ThreatIntelligenceService {
  private threatFeeds: Map<string, ThreatIndicator[]> = new Map();
  private blacklistedIPs: Set<string> = new Set();
  private blacklistedEmails: Set<string> = new Set();

  async checkThreatIndicators(data: {
    ip?: string;
    email?: string;
    domain?: string;
    hash?: string;
  }): Promise<{ isThreat: boolean; level: ThreatLevel; indicators: string[] }> {
    const indicators: string[] = [];
    let maxLevel: ThreatLevel = 'low';

    if (data.ip && this.blacklistedIPs.has(data.ip)) {
      indicators.push('Blacklisted IP');
      maxLevel = 'critical';
    }

    if (data.email && this.blacklistedEmails.has(data.email)) {
      indicators.push('Blacklisted Email');
      maxLevel = 'high';
    }

    if (data.domain) {
      const domainThreat = await this.checkDomainReputation(data.domain);
      if (domainThreat.isThreat) {
        indicators.push(`Suspicious Domain: ${domainThreat.reason}`);
        maxLevel = this.escalateThreatLevel(maxLevel, domainThreat.level);
      }
    }

    return {
      isThreat: indicators.length > 0,
      level: maxLevel,
      indicators
    };
  }

  private async checkDomainReputation(domain: string): Promise<{
    isThreat: boolean;
    level: ThreatLevel;
    reason: string;
  }> {
    const suspiciousTLDs = ['.tk', '.ml', '.ga', '.cf', '.gq'];
    
    if (suspiciousTLDs.some(tld => domain.endsWith(tld))) {
      return {
        isThreat: true,
        level: 'medium',
        reason: 'Suspicious TLD'
      };
    }

    return { isThreat: false, level: 'low', reason: '' };
  }

  private escalateThreatLevel(current: ThreatLevel, new_level: ThreatLevel): ThreatLevel {
    const levels: ThreatLevel[] = ['low', 'medium', 'high', 'critical'];
    const currentIndex = levels.indexOf(current);
    const newIndex = levels.indexOf(new_level);
    return levels[Math.max(currentIndex, newIndex)];
  }

  async updateThreatFeeds(): Promise<void> {
    logger.info('Updating threat intelligence feeds');
    // Implementation for updating threat feeds from external sources
  }

  addToBlacklist(type: 'ip' | 'email', value: string): void {
    if (type === 'ip') {
      this.blacklistedIPs.add(value);
    } else {
      this.blacklistedEmails.add(value);
    }
    logger.info(`Added ${value} to ${type} blacklist`);
  }
}
