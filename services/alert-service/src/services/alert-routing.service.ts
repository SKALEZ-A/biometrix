import { Alert, AlertSeverity, AlertType } from '../../../packages/shared/src/types/alert.types';
import { logger } from '../../../packages/shared/src/utils/logger';

export interface RoutingRule {
  id: string;
  alertType?: AlertType;
  severity?: AlertSeverity;
  assignTo: string[];
  escalationTime?: number;
  priority: number;
}

export class AlertRoutingService {
  private routingRules: RoutingRule[];
  private defaultAssignees: string[];

  constructor() {
    this.routingRules = this.loadRoutingRules();
    this.defaultAssignees = ['default-team@example.com'];
  }

  private loadRoutingRules(): RoutingRule[] {
    return [
      {
        id: 'critical-fraud',
        alertType: AlertType.FRAUD_DETECTED,
        severity: AlertSeverity.CRITICAL,
        assignTo: ['fraud-team-lead@example.com', 'security-team@example.com'],
        escalationTime: 300000,
        priority: 1
      },
      {
        id: 'high-risk-transaction',
        alertType: AlertType.HIGH_RISK_TRANSACTION,
        severity: AlertSeverity.HIGH,
        assignTo: ['risk-analysts@example.com'],
        escalationTime: 600000,
        priority: 2
      },
      {
        id: 'compliance-violation',
        alertType: AlertType.COMPLIANCE_VIOLATION,
        assignTo: ['compliance-team@example.com'],
        escalationTime: 1800000,
        priority: 3
      }
    ];
  }

  async routeAlert(alert: Alert): Promise<string[]> {
    const matchingRules = this.findMatchingRules(alert);

    if (matchingRules.length === 0) {
      logger.info('No matching routing rules, using default assignees', { alertId: alert.id });
      return this.defaultAssignees;
    }

    const highestPriorityRule = matchingRules.reduce((prev, current) =>
      prev.priority < current.priority ? prev : current
    );

    logger.info('Alert routed', {
      alertId: alert.id,
      ruleId: highestPriorityRule.id,
      assignees: highestPriorityRule.assignTo
    });

    if (highestPriorityRule.escalationTime) {
      this.scheduleEscalation(alert, highestPriorityRule.escalationTime);
    }

    return highestPriorityRule.assignTo;
  }

  private findMatchingRules(alert: Alert): RoutingRule[] {
    return this.routingRules.filter(rule => {
      if (rule.alertType && rule.alertType !== alert.type) {
        return false;
      }

      if (rule.severity && rule.severity !== alert.severity) {
        return false;
      }

      return true;
    });
  }

  private scheduleEscalation(alert: Alert, escalationTime: number): void {
    setTimeout(async () => {
      const currentAlert = await this.getAlert(alert.id);

      if (currentAlert && currentAlert.status === 'open') {
        logger.warn('Alert escalation triggered', { alertId: alert.id });
        await this.escalateAlert(alert);
      }
    }, escalationTime);
  }

  private async getAlert(alertId: string): Promise<Alert | null> {
    // Fetch alert from database
    return null;
  }

  private async escalateAlert(alert: Alert): Promise<void> {
    const escalationAssignees = ['management@example.com', 'cto@example.com'];

    logger.info('Escalating alert', {
      alertId: alert.id,
      escalationAssignees
    });

    // Send escalation notifications
  }

  async updateRoutingRule(ruleId: string, updates: Partial<RoutingRule>): Promise<void> {
    const ruleIndex = this.routingRules.findIndex(r => r.id === ruleId);

    if (ruleIndex === -1) {
      throw new Error(`Routing rule not found: ${ruleId}`);
    }

    this.routingRules[ruleIndex] = {
      ...this.routingRules[ruleIndex],
      ...updates
    };

    logger.info('Routing rule updated', { ruleId, updates });
  }

  async addRoutingRule(rule: RoutingRule): Promise<void> {
    this.routingRules.push(rule);
    logger.info('Routing rule added', { ruleId: rule.id });
  }

  async removeRoutingRule(ruleId: string): Promise<void> {
    this.routingRules = this.routingRules.filter(r => r.id !== ruleId);
    logger.info('Routing rule removed', { ruleId });
  }

  getRoutingRules(): RoutingRule[] {
    return this.routingRules;
  }
}
