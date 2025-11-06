import { logger } from '@shared/utils/logger';
import { AuditLog, AuditLogQuery } from '@shared/types/audit.types';
import { v4 as uuidv4 } from 'uuid';

export class AuditService {
  private logs: Map<string, AuditLog> = new Map();

  async log(auditLog: Partial<AuditLog>): Promise<AuditLog> {
    const newLog: AuditLog = {
      id: uuidv4(),
      userId: auditLog.userId!,
      action: auditLog.action!,
      resource: auditLog.resource!,
      resourceId: auditLog.resourceId,
      ipAddress: auditLog.ipAddress,
      userAgent: auditLog.userAgent,
      metadata: auditLog.metadata,
      timestamp: new Date(),
      severity: auditLog.severity || 'info'
    };

    this.logs.set(newLog.id, newLog);
    logger.info('Audit log created', { id: newLog.id, action: newLog.action });
    
    return newLog;
  }

  async getLogs(query: AuditLogQuery): Promise<AuditLog[]> {
    let logs = Array.from(this.logs.values());

    if (query.startDate) {
      logs = logs.filter(log => log.timestamp >= query.startDate!);
    }

    if (query.endDate) {
      logs = logs.filter(log => log.timestamp <= query.endDate!);
    }

    logs = logs.slice(query.offset || 0, (query.offset || 0) + (query.limit || 100));

    return logs;
  }

  async getById(id: string): Promise<AuditLog | null> {
    return this.logs.get(id) || null;
  }

  async getUserAuditTrail(userId: string): Promise<AuditLog[]> {
    return Array.from(this.logs.values())
      .filter(log => log.userId === userId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  async search(query: string, filters: Record<string, any>): Promise<AuditLog[]> {
    let logs = Array.from(this.logs.values());

    if (query) {
      logs = logs.filter(log => 
        log.action.includes(query) || 
        log.resource.includes(query)
      );
    }

    if (filters.userId) {
      logs = logs.filter(log => log.userId === filters.userId);
    }

    if (filters.action) {
      logs = logs.filter(log => log.action === filters.action);
    }

    return logs;
  }

  async export(format: 'json' | 'csv', filters: Record<string, any>): Promise<any> {
    const logs = await this.search('', filters);

    if (format === 'json') {
      return logs;
    }

    // CSV export
    const headers = ['id', 'userId', 'action', 'resource', 'timestamp'];
    const rows = logs.map(log => [
      log.id,
      log.userId,
      log.action,
      log.resource,
      log.timestamp.toISOString()
    ]);

    return { headers, rows };
  }
}
