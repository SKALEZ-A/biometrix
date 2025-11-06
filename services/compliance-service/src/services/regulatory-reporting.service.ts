import { EventEmitter } from 'events';

export interface SuspiciousActivityReport {
  sarId: string;
  filingInstitution: string;
  reportingDate: Date;
  activityDate: Date;
  subjectInformation: {
    name: string;
    address: string;
    identification: string;
    identificationType: string;
    dateOfBirth?: Date;
  };
  suspiciousActivity: {
    type: SARActivityType[];
    description: string;
    amountInvolved: number;
    currency: string;
    transactionIds: string[];
  };
  narrative: string;
  filingReason: string[];
  lawEnforcementContacted: boolean;
  status: 'draft' | 'submitted' | 'accepted' | 'rejected';
  submittedAt?: Date;
  submittedBy?: string;
}

export enum SARActivityType {
  StructuringTransactions = 'structuring_transactions',
  MoneyLaundering = 'money_laundering',
  TerroristFinancing = 'terrorist_financing',
  FraudWireTransfer = 'fraud_wire_transfer',
  CheckFraud = 'check_fraud',
  CreditCardFraud = 'credit_card_fraud',
  IdentityTheft = 'identity_theft',
  ComputerIntrusion = 'computer_intrusion',
  EmbezzlementMisappropriation = 'embezzlement_misappropriation',
  BriberyGratuity = 'bribery_gratuity',
  UnauthorizedElectronicIntrusion = 'unauthorized_electronic_intrusion',
  Other = 'other',
}

export interface CurrencyTransactionReport {
  ctrId: string;
  filingInstitution: string;
  transactionDate: Date;
  filingDate: Date;
  personConductingTransaction: {
    name: string;
    address: string;
    identification: string;
    identificationType: string;
    dateOfBirth?: Date;
    occupation?: string;
  };
  transactionDetails: {
    type: 'deposit' | 'withdrawal' | 'exchange' | 'other';
    amount: number;
    currency: string;
    cashIn: number;
    cashOut: number;
    accountNumber?: string;
  };
  multipleTransactions: boolean;
  aggregatedAmount?: number;
  status: 'draft' | 'submitted' | 'accepted' | 'rejected';
  submittedAt?: Date;
}

export interface ComplianceRule {
  ruleId: string;
  name: string;
  description: string;
  ruleType: 'sar' | 'ctr' | 'aml' | 'kyc' | 'sanctions';
  enabled: boolean;
  conditions: RuleCondition[];
  actions: RuleAction[];
  priority: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface RuleCondition {
  field: string;
  operator: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'contains' | 'in' | 'not_in';
  value: any;
  logicalOperator?: 'AND' | 'OR';
}

export interface RuleAction {
  type: 'generate_sar' | 'generate_ctr' | 'flag_transaction' | 'block_transaction' | 'alert_compliance';
  parameters: Record<string, any>;
}

export interface ComplianceAlert {
  alertId: string;
  ruleId: string;
  ruleName: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  transactionId: string;
  userId: string;
  description: string;
  details: Record<string, any>;
  status: 'open' | 'investigating' | 'resolved' | 'false_positive';
  createdAt: Date;
  assignedTo?: string;
  resolvedAt?: Date;
  resolution?: string;
}

export class RegulatoryReportingService extends EventEmitter {
  private sars: Map<string, SuspiciousActivityReport> = new Map();
  private ctrs: Map<string, CurrencyTransactionReport> = new Map();
  private rules: Map<string, ComplianceRule> = new Map();
  private alerts: Map<string, ComplianceAlert> = new Map();

  constructor() {
    super();
    this.initializeDefaultRules();
  }

  private initializeDefaultRules(): void {
    // SAR Rule: Large cash transactions
    this.addRule({
      ruleId: 'sar_large_cash',
      name: 'Large Cash Transaction',
      description: 'Detect large cash transactions that may require SAR filing',
      ruleType: 'sar',
      enabled: true,
      conditions: [
        {
          field: 'amount',
          operator: 'greater_than',
          value: 10000,
        },
        {
          field: 'paymentMethod',
          operator: 'equals',
          value: 'cash',
          logicalOperator: 'AND',
        },
      ],
      actions: [
        {
          type: 'alert_compliance',
          parameters: {
            severity: 'high',
            message: 'Large cash transaction detected',
          },
        },
      ],
      priority: 1,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    // CTR Rule: Cash transactions over $10,000
    this.addRule({
      ruleId: 'ctr_threshold',
      name: 'CTR Threshold Exceeded',
      description: 'Automatically generate CTR for cash transactions over $10,000',
      ruleType: 'ctr',
      enabled: true,
      conditions: [
        {
          field: 'cashAmount',
          operator: 'greater_than',
          value: 10000,
        },
      ],
      actions: [
        {
          type: 'generate_ctr',
          parameters: {
            autoSubmit: false,
          },
        },
      ],
      priority: 1,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    // SAR Rule: Structuring (multiple transactions just below threshold)
    this.addRule({
      ruleId: 'sar_structuring',
      name: 'Potential Structuring',
      description: 'Detect multiple transactions just below reporting threshold',
      ruleType: 'sar',
      enabled: true,
      conditions: [
        {
          field: 'transactionCount24h',
          operator: 'greater_than',
          value: 3,
        },
        {
          field: 'totalAmount24h',
          operator: 'greater_than',
          value: 10000,
          logicalOperator: 'AND',
        },
        {
          field: 'maxSingleTransaction',
          operator: 'less_than',
          value: 10000,
          logicalOperator: 'AND',
        },
      ],
      actions: [
        {
          type: 'generate_sar',
          parameters: {
            activityType: SARActivityType.StructuringTransactions,
            autoSubmit: false,
          },
        },
        {
          type: 'alert_compliance',
          parameters: {
            severity: 'critical',
            message: 'Potential structuring activity detected',
          },
        },
      ],
      priority: 1,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    // AML Rule: High-risk country transactions
    this.addRule({
      ruleId: 'aml_high_risk_country',
      name: 'High-Risk Country Transaction',
      description: 'Flag transactions involving high-risk countries',
      ruleType: 'aml',
      enabled: true,
      conditions: [
        {
          field: 'country',
          operator: 'in',
          value: ['IR', 'KP', 'SY', 'CU'],  // Example high-risk countries
        },
      ],
      actions: [
        {
          type: 'flag_transaction',
          parameters: {
            flagType: 'high_risk_country',
          },
        },
        {
          type: 'alert_compliance',
          parameters: {
            severity: 'high',
            message: 'Transaction involving high-risk country',
          },
        },
      ],
      priority: 2,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    // KYC Rule: Incomplete customer information
    this.addRule({
      ruleId: 'kyc_incomplete_info',
      name: 'Incomplete KYC Information',
      description: 'Flag customers with incomplete KYC information',
      ruleType: 'kyc',
      enabled: true,
      conditions: [
        {
          field: 'kycStatus',
          operator: 'equals',
          value: 'incomplete',
        },
        {
          field: 'transactionAmount',
          operator: 'greater_than',
          value: 1000,
          logicalOperator: 'AND',
        },
      ],
      actions: [
        {
          type: 'block_transaction',
          parameters: {
            reason: 'Incomplete KYC information',
          },
        },
        {
          type: 'alert_compliance',
          parameters: {
            severity: 'medium',
            message: 'Transaction blocked due to incomplete KYC',
          },
        },
      ],
      priority: 1,
      createdAt: new Date(),
      updatedAt: new Date(),
    });
  }

  /**
   * Create a Suspicious Activity Report
   */
  async createSAR(data: Omit<SuspiciousActivityReport, 'sarId' | 'status'>): Promise<SuspiciousActivityReport> {
    const sarId = this.generateSARId();

    const sar: SuspiciousActivityReport = {
      ...data,
      sarId,
      status: 'draft',
    };

    this.sars.set(sarId, sar);
    this.emit('sar_created', sar);

    return sar;
  }

  /**
   * Submit a Suspicious Activity Report
   */
  async submitSAR(sarId: string, submittedBy: string): Promise<SuspiciousActivityReport> {
    const sar = this.sars.get(sarId);
    if (!sar) {
      throw new Error(`SAR not found: ${sarId}`);
    }

    if (sar.status !== 'draft') {
      throw new Error(`SAR already submitted: ${sarId}`);
    }

    sar.status = 'submitted';
    sar.submittedAt = new Date();
    sar.submittedBy = submittedBy;

    this.sars.set(sarId, sar);
    this.emit('sar_submitted', sar);

    // In production, integrate with FinCEN BSA E-Filing System
    await this.submitToFinCEN(sar);

    return sar;
  }

  /**
   * Create a Currency Transaction Report
   */
  async createCTR(data: Omit<CurrencyTransactionReport, 'ctrId' | 'status'>): Promise<CurrencyTransactionReport> {
    const ctrId = this.generateCTRId();

    const ctr: CurrencyTransactionReport = {
      ...data,
      ctrId,
      status: 'draft',
    };

    this.ctrs.set(ctrId, ctr);
    this.emit('ctr_created', ctr);

    return ctr;
  }

  /**
   * Submit a Currency Transaction Report
   */
  async submitCTR(ctrId: string): Promise<CurrencyTransactionReport> {
    const ctr = this.ctrs.get(ctrId);
    if (!ctr) {
      throw new Error(`CTR not found: ${ctrId}`);
    }

    if (ctr.status !== 'draft') {
      throw new Error(`CTR already submitted: ${ctrId}`);
    }

    ctr.status = 'submitted';
    ctr.submittedAt = new Date();

    this.ctrs.set(ctrId, ctr);
    this.emit('ctr_submitted', ctr);

    // In production, integrate with FinCEN BSA E-Filing System
    await this.submitToFinCEN(ctr);

    return ctr;
  }

  /**
   * Evaluate transaction against compliance rules
   */
  async evaluateTransaction(transaction: Record<string, any>): Promise<{
    violations: ComplianceRule[];
    alerts: ComplianceAlert[];
    actions: RuleAction[];
  }> {
    const violations: ComplianceRule[] = [];
    const alerts: ComplianceAlert[] = [];
    const actions: RuleAction[] = [];

    // Sort rules by priority
    const sortedRules = Array.from(this.rules.values()).sort((a, b) => a.priority - b.priority);

    for (const rule of sortedRules) {
      if (!rule.enabled) continue;

      const isViolation = this.evaluateRuleConditions(rule.conditions, transaction);

      if (isViolation) {
        violations.push(rule);
        actions.push(...rule.actions);

        // Create compliance alert
        const alert = await this.createComplianceAlert({
          ruleId: rule.ruleId,
          ruleName: rule.name,
          severity: this.determineSeverity(rule),
          transactionId: transaction.transactionId,
          userId: transaction.userId,
          description: rule.description,
          details: transaction,
        });

        alerts.push(alert);
      }
    }

    // Execute actions
    for (const action of actions) {
      await this.executeRuleAction(action, transaction);
    }

    return { violations, alerts, actions };
  }

  /**
   * Evaluate rule conditions against transaction data
   */
  private evaluateRuleConditions(conditions: RuleCondition[], data: Record<string, any>): boolean {
    if (conditions.length === 0) return false;

    let result = this.evaluateCondition(conditions[0], data);

    for (let i = 1; i < conditions.length; i++) {
      const condition = conditions[i];
      const conditionResult = this.evaluateCondition(condition, data);

      if (condition.logicalOperator === 'OR') {
        result = result || conditionResult;
      } else {
        // Default to AND
        result = result && conditionResult;
      }
    }

    return result;
  }

  /**
   * Evaluate a single condition
   */
  private evaluateCondition(condition: RuleCondition, data: Record<string, any>): boolean {
    const fieldValue = data[condition.field];

    switch (condition.operator) {
      case 'equals':
        return fieldValue === condition.value;
      case 'not_equals':
        return fieldValue !== condition.value;
      case 'greater_than':
        return Number(fieldValue) > Number(condition.value);
      case 'less_than':
        return Number(fieldValue) < Number(condition.value);
      case 'contains':
        return String(fieldValue).includes(String(condition.value));
      case 'in':
        return Array.isArray(condition.value) && condition.value.includes(fieldValue);
      case 'not_in':
        return Array.isArray(condition.value) && !condition.value.includes(fieldValue);
      default:
        return false;
    }
  }

  /**
   * Execute rule action
   */
  private async executeRuleAction(action: RuleAction, transaction: Record<string, any>): Promise<void> {
    switch (action.type) {
      case 'generate_sar':
        await this.autoGenerateSAR(transaction, action.parameters);
        break;
      case 'generate_ctr':
        await this.autoGenerateCTR(transaction, action.parameters);
        break;
      case 'flag_transaction':
        this.emit('transaction_flagged', { transaction, parameters: action.parameters });
        break;
      case 'block_transaction':
        this.emit('transaction_blocked', { transaction, parameters: action.parameters });
        break;
      case 'alert_compliance':
        // Alert already created in evaluateTransaction
        break;
    }
  }

  /**
   * Auto-generate SAR from transaction
   */
  private async autoGenerateSAR(transaction: Record<string, any>, parameters: Record<string, any>): Promise<void> {
    const sar = await this.createSAR({
      filingInstitution: 'Fraud Prevention System',
      reportingDate: new Date(),
      activityDate: new Date(transaction.timestamp),
      subjectInformation: {
        name: transaction.userName || 'Unknown',
        address: transaction.userAddress || 'Unknown',
        identification: transaction.userId,
        identificationType: 'User ID',
      },
      suspiciousActivity: {
        type: [parameters.activityType || SARActivityType.Other],
        description: `Suspicious activity detected: ${transaction.description || 'N/A'}`,
        amountInvolved: transaction.amount,
        currency: transaction.currency || 'USD',
        transactionIds: [transaction.transactionId],
      },
      narrative: `Automated SAR generated based on compliance rule violation. Transaction details: ${JSON.stringify(transaction)}`,
      filingReason: ['Automated detection', parameters.reason || 'Rule violation'],
      lawEnforcementContacted: false,
    });

    if (parameters.autoSubmit) {
      await this.submitSAR(sar.sarId, 'system');
    }
  }

  /**
   * Auto-generate CTR from transaction
   */
  private async autoGenerateCTR(transaction: Record<string, any>, parameters: Record<string, any>): Promise<void> {
    const ctr = await this.createCTR({
      filingInstitution: 'Fraud Prevention System',
      transactionDate: new Date(transaction.timestamp),
      filingDate: new Date(),
      personConductingTransaction: {
        name: transaction.userName || 'Unknown',
        address: transaction.userAddress || 'Unknown',
        identification: transaction.userId,
        identificationType: 'User ID',
      },
      transactionDetails: {
        type: transaction.transactionType || 'other',
        amount: transaction.amount,
        currency: transaction.currency || 'USD',
        cashIn: transaction.cashIn || 0,
        cashOut: transaction.cashOut || 0,
        accountNumber: transaction.accountNumber,
      },
      multipleTransactions: transaction.multipleTransactions || false,
      aggregatedAmount: transaction.aggregatedAmount,
    });

    if (parameters.autoSubmit) {
      await this.submitCTR(ctr.ctrId);
    }
  }

  /**
   * Create compliance alert
   */
  private async createComplianceAlert(
    data: Omit<ComplianceAlert, 'alertId' | 'status' | 'createdAt'>
  ): Promise<ComplianceAlert> {
    const alertId = this.generateAlertId();

    const alert: ComplianceAlert = {
      ...data,
      alertId,
      status: 'open',
      createdAt: new Date(),
    };

    this.alerts.set(alertId, alert);
    this.emit('compliance_alert_created', alert);

    return alert;
  }

  /**
   * Add compliance rule
   */
  addRule(rule: ComplianceRule): void {
    this.rules.set(rule.ruleId, rule);
    this.emit('rule_added', rule);
  }

  /**
   * Update compliance rule
   */
  updateRule(ruleId: string, updates: Partial<ComplianceRule>): ComplianceRule {
    const rule = this.rules.get(ruleId);
    if (!rule) {
      throw new Error(`Rule not found: ${ruleId}`);
    }

    const updatedRule = {
      ...rule,
      ...updates,
      updatedAt: new Date(),
    };

    this.rules.set(ruleId, updatedRule);
    this.emit('rule_updated', updatedRule);

    return updatedRule;
  }

  /**
   * Delete compliance rule
   */
  deleteRule(ruleId: string): void {
    const rule = this.rules.get(ruleId);
    if (!rule) {
      throw new Error(`Rule not found: ${ruleId}`);
    }

    this.rules.delete(ruleId);
    this.emit('rule_deleted', rule);
  }

  /**
   * Get all SARs
   */
  getAllSARs(): SuspiciousActivityReport[] {
    return Array.from(this.sars.values());
  }

  /**
   * Get all CTRs
   */
  getAllCTRs(): CurrencyTransactionReport[] {
    return Array.from(this.ctrs.values());
  }

  /**
   * Get all compliance alerts
   */
  getAllAlerts(): ComplianceAlert[] {
    return Array.from(this.alerts.values());
  }

  /**
   * Get alerts by status
   */
  getAlertsByStatus(status: ComplianceAlert['status']): ComplianceAlert[] {
    return Array.from(this.alerts.values()).filter(alert => alert.status === status);
  }

  /**
   * Resolve compliance alert
   */
  resolveAlert(alertId: string, resolution: string, resolvedBy: string): ComplianceAlert {
    const alert = this.alerts.get(alertId);
    if (!alert) {
      throw new Error(`Alert not found: ${alertId}`);
    }

    alert.status = 'resolved';
    alert.resolvedAt = new Date();
    alert.resolution = resolution;
    alert.assignedTo = resolvedBy;

    this.alerts.set(alertId, alert);
    this.emit('alert_resolved', alert);

    return alert;
  }

  /**
   * Submit report to FinCEN (simulated)
   */
  private async submitToFinCEN(report: SuspiciousActivityReport | CurrencyTransactionReport): Promise<void> {
    // In production, integrate with FinCEN BSA E-Filing System
    console.log('Submitting report to FinCEN:', report);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Update status to accepted
    if ('sarId' in report) {
      report.status = 'accepted';
      this.sars.set(report.sarId, report);
    } else {
      report.status = 'accepted';
      this.ctrs.set(report.ctrId, report);
    }
  }

  private generateSARId(): string {
    return `SAR-${Date.now()}-${Math.random().toString(36).substring(2, 9).toUpperCase()}`;
  }

  private generateCTRId(): string {
    return `CTR-${Date.now()}-${Math.random().toString(36).substring(2, 9).toUpperCase()}`;
  }

  private generateAlertId(): string {
    return `ALERT-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  private determineSeverity(rule: ComplianceRule): ComplianceAlert['severity'] {
    if (rule.priority === 1) return 'critical';
    if (rule.priority === 2) return 'high';
    if (rule.priority === 3) return 'medium';
    return 'low';
  }
}
