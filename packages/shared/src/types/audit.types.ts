export type AuditSeverity = 'info' | 'warning' | 'error' | 'critical';

export interface AuditLog {
  id: string;
  userId: string;
  action: string;
  resource: string;
  resourceId?: string;
  ipAddress?: string;
  userAgent?: string;
  metadata?: Record<string, any>;
  timestamp: Date;
  severity: AuditSeverity;
}

export interface AuditLogQuery {
  startDate?: Date;
  endDate?: Date;
  userId?: string;
  action?: string;
  resource?: string;
  severity?: AuditSeverity;
  limit?: number;
  offset?: number;
}

export interface AuditTrail {
  userId: string;
  logs: AuditLog[];
  totalCount: number;
  period: {
    start: Date;
    end: Date;
  };
}

export interface ComplianceReport {
  id: string;
  reportType: string;
  period: {
    start: Date;
    end: Date;
  };
  totalEvents: number;
  criticalEvents: number;
  summary: Record<string, any>;
  generatedAt: Date;
}
