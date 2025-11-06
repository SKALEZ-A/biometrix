import { Request, Response } from 'express';
import { AnalyticsService } from '../services/analytics.service';
import { logger } from '@shared/utils/logger';

export class AnalyticsController {
  private analyticsService: AnalyticsService;

  constructor() {
    this.analyticsService = new AnalyticsService();
  }

  async trackEvent(req: Request, res: Response): Promise<void> {
    try {
      const event = req.body;
      await this.analyticsService.trackEvent(event);
      res.status(200).json({ success: true });
    } catch (error) {
      logger.error('Failed to track event', { error });
      res.status(500).json({ error: 'Failed to track event' });
    }
  }

  async getMetrics(req: Request, res: Response): Promise<void> {
    try {
      const { startDate, endDate, metricType } = req.query;
      const metrics = await this.analyticsService.getMetrics(
        new Date(startDate as string),
        new Date(endDate as string),
        metricType as string
      );
      res.status(200).json(metrics);
    } catch (error) {
      logger.error('Failed to get metrics', { error });
      res.status(500).json({ error: 'Failed to get metrics' });
    }
  }

  async getDashboard(req: Request, res: Response): Promise<void> {
    try {
      const dashboard = await this.analyticsService.getDashboardMetrics();
      res.status(200).json(dashboard);
    } catch (error) {
      logger.error('Failed to get dashboard', { error });
      res.status(500).json({ error: 'Failed to get dashboard' });
    }
  }

  async getFraudStats(req: Request, res: Response): Promise<void> {
    try {
      const { period } = req.query;
      const stats = await this.analyticsService.getFraudStatistics(period as string);
      res.status(200).json(stats);
    } catch (error) {
      logger.error('Failed to get fraud stats', { error });
      res.status(500).json({ error: 'Failed to get fraud stats' });
    }
  }
}
