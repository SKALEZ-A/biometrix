export interface Alert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  description: string;
  entityId: string;
  entityType: string;
  status: AlertStatus;
  assignedTo?: string;
  createdAt: Date;
  updatedAt: Date;
  resolvedAt?: Date;
  metadata?: Record<string, any>;
}

export enum AlertType {
  FRAUD_DETECTED = 'fraud_detected',
  HIGH_RISK_TRANSACTION = 'high_risk_transaction',
  SUSPICIOUS_PATTERN = 'suspicious_pattern',
  ACCOUNT_TAKEOVER = 'account_takeover',
  VELOCITY_BREACH = 'velocity_breach',
  BIOMETRIC_MISMATCH = 'biometric_mismatch',
  COMPLIANCE_VIOLATION = 'compliance_violation',
  SYSTEM_ANOMALY = 'system_anomaly'
}

export enum AlertSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum AlertStatus {
  OPEN = 'open',
  INVESTIGATING = 'investigating',
  RESOLVED = 'resolved',
  FALSE_POSITIVE = 'false_positive',
  ESCALATED = 'escalated'
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  type: AlertType;
  severity: AlertSeverity;
  conditions: AlertCondition[];
  actions: AlertAction[];
  enabled: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface AlertCondition {
  field: string;
  operator: ConditionOperator;
  value: any;
  logicalOperator?: LogicalOperator;
}

export enum ConditionOperator {
  EQUALS = 'equals',
  NOT_EQUALS = 'not_equals',
  GREATER_THAN = 'greater_than',
  LESS_THAN = 'less_than',
  CONTAINS = 'contains',
  IN = 'in',
  BETWEEN = 'between'
}

export enum LogicalOperator {
  AND = 'and',
  OR = 'or'
}

export interface AlertAction {
  type: ActionType;
  config: Record<string, any>;
}

export enum ActionType {
  SEND_EMAIL = 'send_email',
  SEND_SMS = 'send_sms',
  SEND_PUSH = 'send_push',
  CREATE_TICKET = 'create_ticket',
  BLOCK_TRANSACTION = 'block_transaction',
  LOCK_ACCOUNT = 'lock_account',
  WEBHOOK = 'webhook'
}
