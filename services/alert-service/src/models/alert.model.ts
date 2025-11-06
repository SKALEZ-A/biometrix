export interface Alert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  status: AlertStatus;
  title: string;
  description: string;
  source: AlertSource;
  entityType: string;
  entityId: string;
  metadata: AlertMetadata;
  assignedTo?: string;
  resolvedBy?: string;
  resolvedAt?: Date;
  resolution?: string;
  createdAt: Date;
  updatedAt: Date;
}

export enum AlertType {
  FRAUD_DETECTED = 'FRAUD_DETECTED',
  HIGH_RISK_TRANSACTION = 'HIGH_RISK_TRANSACTION',
  SUSPICIOUS_ACTIVITY = 'SUSPICIOUS_ACTIVITY',
  VELOCITY_BREACH = 'VELOCITY_BREACH',
  BIOMETRIC_MISMATCH = 'BIOMETRIC_MISMATCH',
  COMPLIANCE_VIOLATION = 'COMPLIANCE_VIOLATION',
  SYSTEM_ANOMALY = 'SYSTEM_ANOMALY',
  CHARGEBACK_RISK = 'CHARGEBACK_RISK'
}

export enum AlertSeverity {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export enum AlertStatus {
  NEW = 'NEW',
  ACKNOWLEDGED = 'ACKNOWLEDGED',
  IN_PROGRESS = 'IN_PROGRESS',
  RESOLVED = 'RESOLVED',
  FALSE_POSITIVE = 'FALSE_POSITIVE',
  ESCALATED = 'ESCALATED'
}

export enum AlertSource {
  FRAUD_DETECTION = 'FRAUD_DETECTION',
  BIOMETRIC_SERVICE = 'BIOMETRIC_SERVICE',
  TRANSACTION_SERVICE = 'TRANSACTION_SERVICE',
  COMPLIANCE_SERVICE = 'COMPLIANCE_SERVICE',
  ML_MODEL = 'ML_MODEL',
  MANUAL = 'MANUAL'
}

export interface AlertMetadata {
  riskScore?: number;
  transactionId?: string;
  userId?: string;
  merchantId?: string;
  amount?: number;
  location?: string;
  deviceId?: string;
  ipAddress?: string;
  additionalInfo?: any;
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
}

export enum ConditionOperator {
  EQUALS = 'EQUALS',
  NOT_EQUALS = 'NOT_EQUALS',
  GREATER_THAN = 'GREATER_THAN',
  LESS_THAN = 'LESS_THAN',
  CONTAINS = 'CONTAINS',
  IN = 'IN'
}

export interface AlertAction {
  type: ActionType;
  config: any;
}

export enum ActionType {
  SEND_EMAIL = 'SEND_EMAIL',
  SEND_SMS = 'SEND_SMS',
  WEBHOOK = 'WEBHOOK',
  BLOCK_TRANSACTION = 'BLOCK_TRANSACTION',
  ESCALATE = 'ESCALATE'
}
