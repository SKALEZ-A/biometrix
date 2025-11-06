import { Request, Response } from 'express';
import { AuditService } from '../services/audit.service';
import { logger } from '@shared/utils/logger';

export class AuditController {
  private auditService: AuditService;

  constructor() {
    this.auditService = new AuditService();
  }

  async logEvent(req: Request, res: Response): Promise<void> {
    try {
      const auditLog = req.body;
      const result = await this.auditService.log(auditLog);
      res.status(201).json(result);
    } catch (error) {
      logger.error('Failed to log audit event', { error });
      res.status(500).json({ error: 'Failed to log audit event' });
    }
  }

  async getLogs(req: Request, res: Response): Promise<void> {
    try {
      const { startDate, endDate, limit, offset } = req.query;
      const logs = await this.auditService.getLogs({
        startDate: startDate ? new Date(startDate as string) : undefined,
        endDate: endDate ? new Date(endDate as string) : undefined,
        limit: limit ? parseInt(limit as string) : 100,
        offset: offset ? parseInt(offset as string) : 0
      });
      res.status(200).json(logs);
    } catch (error) {
      logger.error('Failed to get audit logs', { error });
      res.status(500).json({ error: 'Failed to get audit logs' });
    }
  }

  async getLogById(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const log = await this.auditService.getById(id);
      res.status(200).json(log);
    } catch (error) {
      logger.error('Failed to get audit log', { error });
      res.status(500).json({ error: 'Failed to get audit log' });
    }
  }

  async getUserAuditTrail(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;
      const trail = await this.auditService.getUserAuditTrail(userId);
      res.status(200).json(trail);
    } catch (error) {
      logger.error('Failed to get user audit trail', { error });
      res.status(500).json({ error: 'Failed to get user audit trail' });
    }
  }

  async searchLogs(req: Request, res: Response): Promise<void> {
    try {
      const { query, filters } = req.query;
      const results = await this.auditService.search(
        query as string,
        filters ? JSON.parse(filters as string) : {}
      );
      res.status(200).json(results);
    } catch (error) {
      logger.error('Failed to search audit logs', { error });
      res.status(500).json({ error: 'Failed to search audit logs' });
    }
  }

  async exportLogs(req: Request, res: Response): Promise<void> {
    try {
      const { format, filters } = req.body;
      const exported = await this.auditService.export(format, filters);
      res.status(200).json(exported);
    } catch (error) {
      logger.error('Failed to export audit logs', { error });
      res.status(500).json({ error: 'Failed to export audit logs' });
    }
  }
}
