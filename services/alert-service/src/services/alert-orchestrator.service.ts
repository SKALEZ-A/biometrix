import { Alert, AlertRule, AlertChannel, AlertPriority, AlertStatus } from '@shared/types/alert.types';
import { Logger } from '@shared/utils/logger';
import { EventEmitter } from 'events';

export class AlertOrchestrator Service extends EventEmitter {
  private readonly logger = new Logger('AlertOrchestratorService');
  private rules: Map<string, AlertRule> = new Map();
  private activeAlerts: Map<string, Alert> = new Map();
  private alertHistory: Alert[] = [];
  private channels: Map<string, AlertChannel> = new Map();
  private rateLimits: Map<string, number[]> = new Map();

  constructor() {
    super();
    this.initializeDefaultRules();
    this.initializeChannels();
  }

  private initializeDefaultRules(): void {
    this.addRule({
      id: 'high_value_transaction',
      name: 'High Value Transaction Alert',
      description: 'Triggers when transaction exceeds threshold',
      condition: {
        type: 'threshold',
        metric: 'transaction_amount',
        operator: '>',
        value: 50000
      },
      priority: 'HIGH',
      channels: ['email', 'sms', 'slack'],
      enabled: true,
      cooldownMinutes: 5
    });

    this.addRule({
      id: 'fraud_score_critical',
      name: 'Critical Fraud Score',
      description: 'Triggers when fraud score exceeds critical threshold',
      condition: {
        type: 'threshold',
        metric: 'fraud_score',
        operator: '>=',
        value: 0.9
      },
      priority: 'CRITICAL',
      channels: ['email', 'sms', 'slack', 'pagerduty'],
      enabled: true,
      cooldownMinutes: 0
    });

    this.addRule({
      id: 'velocity_violation',
      name: 'Velocity Rule Violation',
      description: 'Triggers when velocity limits are exceeded',
      condition: {
        type: 'composite',
        operator: 'OR',
        conditions: [
          { type: 'threshold', metric: 'txn_count_1h', operator: '>', value: 10 },
          { type: 'threshold', metric: 'txn_amount_1h', operator: '>', value: 100000 }
        ]
      },
      priority: 'HIGH',
      channels: ['email', 'slack'],
      enabled: true,
      cooldownMinutes: 15
    });

    this.addRule({
      id: 'watchlist_match',
      name: 'Watchlist Match Detected',
      description: 'Triggers when entity matches sanctions or PEP list',
      condition: {
        type: 'boolean',
        metric: 'watchlist_match',
        value: true
      },
      priority: 'CRITICAL',
      channels: ['email', 'sms', 'slack'],
      enabled: true,
      cooldownMinutes: 0
    });

    this.addRule({
      id: 'biometric_mismatch',
      name: 'Biometric Authentication Failure',
      description: 'Triggers on biometric verification failure',
      condition: {
        type: 'threshold',
        metric: 'biometric_confidence',
        operator: '<',
        value: 0.7
      },
      priority: 'MEDIUM',
      channels: ['email', 'slack'],
      enabled: true,
      cooldownMinutes: 10
    });
  }

  private initializeChannels(): void {
    this.channels.set('email', {
      id: 'email',
      name: 'Email',
      type: 'email',
      enabled: true,
      config: {
        from: 'alerts@frauddetection.com',
        recipients: ['security@company.com', 'fraud-team@company.com']
      }
    });

    this.channels.set('sms', {
      id: 'sms',
      name: 'SMS',
      type: 'sms',
      enabled: true,
      config: {
        phoneNumbers: ['+1234567890', '+0987654321']
      }
    });

    this.channels.set('slack', {
      id: 'slack',
      name: 'Slack',
      type: 'webhook',
      enabled: true,
      config: {
        webhookUrl: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        channel: '#fraud-alerts'
      }
    });

    this.channels.set('pagerduty', {
      id: 'pagerduty',
      name: 'PagerDuty',
      type: 'webhook',
      enabled: true,
      config: {
        integrationKey: 'YOUR_PAGERDUTY_KEY',
        severity: 'critical'
      }
    });
  }

  async evaluateEvent(event: any): Promise<Alert[]> {
    const triggeredAlerts: Alert[] = [];

    for (const [ruleId, rule] of this.rules.entries()) {
      if (!rule.enabled) continue;

      if (this.isInCooldown(ruleId, event.userId)) {
        continue;
      }

      const isTriggered = this.evaluateCondition(rule.condition, event);

      if (isTriggered) {
        const alert = await this.createAlert(rule, event);
        triggeredAlerts.push(alert);
        
        this.updateCooldown(ruleId, event.userId);
      }
    }

    return triggeredAlerts;
  }

  private evaluateCondition(condition: any, event: any): boolean {
    switch (condition.type) {
      case 'threshold':
        return this.evaluateThreshold(condition, event);
      
      case 'boolean':
        return event[condition.metric] === condition.value;
      
      case 'composite':
        return this.evaluateComposite(condition, event);
      
      case 'pattern':
        return this.evaluatePattern(condition, event);
      
      default:
        return false;
    }
  }

  private evaluateThreshold(condition: any, event: any): boolean {
    const value = event[condition.metric];
    const threshold = condition.value;

    switch (condition.operator) {
      case '>':
        return value > threshold;
      case '>=':
        return value >= threshold;
      case '<':
        return value < threshold;
      case '<=':
        return value <= threshold;
      case '==':
        return value === threshold;
      case '!=':
        return value !== threshold;
      default:
        return false;
    }
  }

  private evaluateComposite(condition: any, event: any): boolean {
    const results = condition.conditions.map((cond: any) => 
      this.evaluateCondition(cond, event)
    );

    switch (condition.operator) {
      case 'AND':
        return results.every((r: boolean) => r);
      case 'OR':
        return results.some((r: boolean) => r);
      case 'NOT':
        return !results[0];
      default:
        return false;
    }
  }

  private evaluatePattern(condition: any, event: any): boolean {
    const pattern = new RegExp(condition.pattern);
    const value = event[condition.metric];
    return pattern.test(String(value));
  }

  private async createAlert(rule: AlertRule, event: any): Promise<Alert> {
    const alertId = `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const alert: Alert = {
      id: alertId,
      ruleId: rule.id,
      ruleName: rule.name,
      priority: rule.priority,
      status: 'OPEN',
      title: this.generateAlertTitle(rule, event),
      description: this.generateAlertDescription(rule, event),
      eventData: event,
      channels: rule.channels,
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    this.activeAlerts.set(alertId, alert);
    this.alertHistory.push(alert);

    await this.dispatchAlert(alert);

    this.emit('alert:created', alert);

    return alert;
  }

  private generateAlertTitle(rule: AlertRule, event: any): string {
    const templates: Record<string, string> = {
      'high_value_transaction': `High Value Transaction: $${event.transaction_amount}`,
      'fraud_score_critical': `Critical Fraud Score: ${(event.fraud_score * 100).toFixed(1)}%`,
      'velocity_violation': `Velocity Limit Exceeded for User ${event.userId}`,
      'watchlist_match': `Watchlist Match: ${event.matchedName}`,
      'biometric_mismatch': `Biometric Verification Failed for User ${event.userId}`
    };

    return templates[rule.id] || rule.name;
  }

  private generateAlertDescription(rule: AlertRule, event: any): string {
    let description = `${rule.description}\n\n`;
    description += `Event Details:\n`;
    
    for (const [key, value] of Object.entries(event)) {
      if (typeof value !== 'object') {
        description += `- ${key}: ${value}\n`;
      }
    }

    return description;
  }

  private async dispatchAlert(alert: Alert): Promise<void> {
    const dispatchPromises = alert.channels.map(channelId => 
      this.sendToChannel(channelId, alert)
    );

    try {
      await Promise.all(dispatchPromises);
      this.logger.info(`Alert ${alert.id} dispatched to ${alert.channels.length} channels`);
    } catch (error) {
      this.logger.error(`Failed to dispatch alert ${alert.id}`, error);
    }
  }

  private async sendToChannel(channelId: string, alert: Alert): Promise<void> {
    const channel = this.channels.get(channelId);
    
    if (!channel || !channel.enabled) {
      return;
    }

    if (!this.checkRateLimit(channelId)) {
      this.logger.warn(`Rate limit exceeded for channel ${channelId}`);
      return;
    }

    try {
      switch (channel.type) {
        case 'email':
          await this.sendEmail(channel, alert);
          break;
        case 'sms':
          await this.sendSMS(channel, alert);
          break;
        case 'webhook':
          await this.sendWebhook(channel, alert);
          break;
        default:
          this.logger.warn(`Unknown channel type: ${channel.type}`);
      }

      this.recordChannelUsage(channelId);
    } catch (error) {
      this.logger.error(`Failed to send alert to channel ${channelId}`, error);
    }
  }

  private async sendEmail(channel: AlertChannel, alert: Alert): Promise<void> {
    const emailContent = {
      from: channel.config.from,
      to: channel.config.recipients,
      subject: `[${alert.priority}] ${alert.title}`,
      body: this.formatEmailBody(alert)
    };

    this.logger.info(`Sending email alert: ${alert.id}`);
  }

  private formatEmailBody(alert: Alert): string {
    return `
      <html>
        <body>
          <h2>${alert.title}</h2>
          <p><strong>Priority:</strong> ${alert.priority}</p>
          <p><strong>Status:</strong> ${alert.status}</p>
          <p><strong>Created:</strong> ${new Date(alert.createdAt).toISOString()}</p>
          <hr>
          <h3>Description</h3>
          <pre>${alert.description}</pre>
          <hr>
          <h3>Event Data</h3>
          <pre>${JSON.stringify(alert.eventData, null, 2)}</pre>
        </body>
      </html>
    `;
  }

  private async sendSMS(channel: AlertChannel, alert: Alert): Promise<void> {
    const smsContent = {
      to: channel.config.phoneNumbers,
      message: `[${alert.priority}] ${alert.title.substring(0, 100)}`
    };

    this.logger.info(`Sending SMS alert: ${alert.id}`);
  }

  private async sendWebhook(channel: AlertChannel, alert: Alert): Promise<void> {
    const payload = {
      alert_id: alert.id,
      priority: alert.priority,
      title: alert.title,
      description: alert.description,
      status: alert.status,
      created_at: alert.createdAt,
      event_data: alert.eventData
    };

    this.logger.info(`Sending webhook alert to ${channel.name}: ${alert.id}`);
  }

  private isInCooldown(ruleId: string, userId: string): boolean {
    const rule = this.rules.get(ruleId);
    if (!rule || rule.cooldownMinutes === 0) return false;

    const cooldownKey = `${ruleId}:${userId}`;
    const lastTrigger = this.rateLimits.get(cooldownKey)?.[0];

    if (!lastTrigger) return false;

    const cooldownMs = rule.cooldownMinutes * 60 * 1000;
    return (Date.now() - lastTrigger) < cooldownMs;
  }

  private updateCooldown(ruleId: string, userId: string): void {
    const cooldownKey = `${ruleId}:${userId}`;
    this.rateLimits.set(cooldownKey, [Date.now()]);
  }

  private checkRateLimit(channelId: string): boolean {
    const limit = 10;
    const windowMs = 60 * 1000;

    const timestamps = this.rateLimits.get(channelId) || [];
    const now = Date.now();
    const recentTimestamps = timestamps.filter(ts => (now - ts) < windowMs);

    return recentTimestamps.length < limit;
  }

  private recordChannelUsage(channelId: string): void {
    const timestamps = this.rateLimits.get(channelId) || [];
    timestamps.push(Date.now());
    
    const windowMs = 60 * 1000;
    const now = Date.now();
    const recentTimestamps = timestamps.filter(ts => (now - ts) < windowMs);
    
    this.rateLimits.set(channelId, recentTimestamps);
  }

  async acknowledgeAlert(alertId: string, acknowledgedBy: string): Promise<void> {
    const alert = this.activeAlerts.get(alertId);
    
    if (!alert) {
      throw new Error(`Alert ${alertId} not found`);
    }

    alert.status = 'ACKNOWLEDGED';
    alert.acknowledgedBy = acknowledgedBy;
    alert.acknowledgedAt = Date.now();
    alert.updatedAt = Date.now();

    this.emit('alert:acknowledged', alert);
    this.logger.info(`Alert ${alertId} acknowledged by ${acknowledgedBy}`);
  }

  async resolveAlert(alertId: string, resolvedBy: string, resolution: string): Promise<void> {
    const alert = this.activeAlerts.get(alertId);
    
    if (!alert) {
      throw new Error(`Alert ${alertId} not found`);
    }

    alert.status = 'RESOLVED';
    alert.resolvedBy = resolvedBy;
    alert.resolvedAt = Date.now();
    alert.resolution = resolution;
    alert.updatedAt = Date.now();

    this.activeAlerts.delete(alertId);

    this.emit('alert:resolved', alert);
    this.logger.info(`Alert ${alertId} resolved by ${resolvedBy}`);
  }

  async escalateAlert(alertId: string, escalatedBy: string, reason: string): Promise<void> {
    const alert = this.activeAlerts.get(alertId);
    
    if (!alert) {
      throw new Error(`Alert ${alertId} not found`);
    }

    const priorityLevels: AlertPriority[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
    const currentIndex = priorityLevels.indexOf(alert.priority);
    
    if (currentIndex < priorityLevels.length - 1) {
      alert.priority = priorityLevels[currentIndex + 1];
    }

    alert.escalatedBy = escalatedBy;
    alert.escalatedAt = Date.now();
    alert.escalationReason = reason;
    alert.updatedAt = Date.now();

    await this.dispatchAlert(alert);

    this.emit('alert:escalated', alert);
    this.logger.info(`Alert ${alertId} escalated to ${alert.priority} by ${escalatedBy}`);
  }

  addRule(rule: AlertRule): void {
    this.rules.set(rule.id, rule);
    this.logger.info(`Added alert rule: ${rule.name}`);
  }

  removeRule(ruleId: string): boolean {
    const deleted = this.rules.delete(ruleId);
    if (deleted) {
      this.logger.info(`Removed alert rule: ${ruleId}`);
    }
    return deleted;
  }

  updateRule(ruleId: string, updates: Partial<AlertRule>): void {
    const rule = this.rules.get(ruleId);
    if (rule) {
      Object.assign(rule, updates);
      this.logger.info(`Updated alert rule: ${ruleId}`);
    }
  }

  getActiveAlerts(filters?: { priority?: AlertPriority; status?: AlertStatus }): Alert[] {
    let alerts = Array.from(this.activeAlerts.values());

    if (filters?.priority) {
      alerts = alerts.filter(a => a.priority === filters.priority);
    }

    if (filters?.status) {
      alerts = alerts.filter(a => a.status === filters.status);
    }

    return alerts.sort((a, b) => b.createdAt - a.createdAt);
  }

  getAlertHistory(limit: number = 100): Alert[] {
    return this.alertHistory
      .slice(-limit)
      .sort((a, b) => b.createdAt - a.createdAt);
  }

  getAlertStatistics(timeWindowMs: number = 24 * 60 * 60 * 1000): any {
    const now = Date.now();
    const recentAlerts = this.alertHistory.filter(a => (now - a.createdAt) < timeWindowMs);

    const byPriority = recentAlerts.reduce((acc, alert) => {
      acc[alert.priority] = (acc[alert.priority] || 0) + 1;
      return acc;
    }, {} as Record<AlertPriority, number>);

    const byStatus = recentAlerts.reduce((acc, alert) => {
      acc[alert.status] = (acc[alert.status] || 0) + 1;
      return acc;
    }, {} as Record<AlertStatus, number>);

    const byRule = recentAlerts.reduce((acc, alert) => {
      acc[alert.ruleId] = (acc[alert.ruleId] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const avgResolutionTime = this.calculateAverageResolutionTime(recentAlerts);

    return {
      totalAlerts: recentAlerts.length,
      activeAlerts: this.activeAlerts.size,
      byPriority,
      byStatus,
      byRule,
      avgResolutionTimeMs: avgResolutionTime,
      timeWindowMs
    };
  }

  private calculateAverageResolutionTime(alerts: Alert[]): number {
    const resolvedAlerts = alerts.filter(a => a.status === 'RESOLVED' && a.resolvedAt);
    
    if (resolvedAlerts.length === 0) return 0;

    const totalTime = resolvedAlerts.reduce((sum, alert) => {
      return sum + (alert.resolvedAt! - alert.createdAt);
    }, 0);

    return totalTime / resolvedAlerts.length;
  }

  async testChannel(channelId: string): Promise<boolean> {
    const channel = this.channels.get(channelId);
    
    if (!channel) {
      throw new Error(`Channel ${channelId} not found`);
    }

    const testAlert: Alert = {
      id: 'test-alert',
      ruleId: 'test',
      ruleName: 'Test Alert',
      priority: 'LOW',
      status: 'OPEN',
      title: 'Test Alert',
      description: 'This is a test alert to verify channel configuration',
      eventData: {},
      channels: [channelId],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    try {
      await this.sendToChannel(channelId, testAlert);
      this.logger.info(`Test alert sent successfully to channel ${channelId}`);
      return true;
    } catch (error) {
      this.logger.error(`Test alert failed for channel ${channelId}`, error);
      return false;
    }
  }

  addChannel(channel: AlertChannel): void {
    this.channels.set(channel.id, channel);
    this.logger.info(`Added alert channel: ${channel.name}`);
  }

  removeChannel(channelId: string): boolean {
    const deleted = this.channels.delete(channelId);
    if (deleted) {
      this.logger.info(`Removed alert channel: ${channelId}`);
    }
    return deleted;
  }

  getChannels(): AlertChannel[] {
    return Array.from(this.channels.values());
  }

  getRules(): AlertRule[] {
    return Array.from(this.rules.values());
  }
}
