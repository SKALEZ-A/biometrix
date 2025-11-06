import { Request, Response } from 'express';
import { ReportService } from '../services/report.service';
import { logger } from '@shared/utils/logger';

export class ReportController {
  private reportService: ReportService;

  constructor() {
    this.reportService = new ReportService();
  }

  async generateReport(req: Request, res: Response): Promise<void> {
    try {
      const { type, format, parameters } = req.body;
      const userId = (req as any).user.id;
      
      const report = await this.reportService.generate(type, format, parameters, userId);
      res.status(202).json(report);
    } catch (error) {
      logger.error('Failed to generate report', { error });
      res.status(500).json({ error: 'Failed to generate report' });
    }
  }

  async getReport(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const report = await this.reportService.getById(id);
      
      if (!report) {
        res.status(404).json({ error: 'Report not found' });
        return;
      }
      
      res.status(200).json(report);
    } catch (error) {
      logger.error('Failed to get report', { error });
      res.status(500).json({ error: 'Failed to get report' });
    }
  }

  async listReports(req: Request, res: Response): Promise<void> {
    try {
      const { type, status } = req.query;
      const reports = await this.reportService.list({
        type: type as any,
        status: status as string
      });
      res.status(200).json(reports);
    } catch (error) {
      logger.error('Failed to list reports', { error });
      res.status(500).json({ error: 'Failed to list reports' });
    }
  }

  async exportReport(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const { format } = req.body;
      
      // Implementation for exporting report in different format
      res.status(200).json({ message: 'Export initiated' });
    } catch (error) {
      logger.error('Failed to export report', { error });
      res.status(500).json({ error: 'Failed to export report' });
    }
  }

  async getScheduledReports(req: Request, res: Response): Promise<void> {
    try {
      // Implementation for getting scheduled reports
      res.status(200).json([]);
    } catch (error) {
      logger.error('Failed to get scheduled reports', { error });
      res.status(500).json({ error: 'Failed to get scheduled reports' });
    }
  }
}
