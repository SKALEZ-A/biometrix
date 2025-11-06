import { Request, Response } from 'express';
import { NotificationService } from '../services/notification.service';
import { logger } from '@shared/utils/logger';

export class NotificationController {
  private notificationService: NotificationService;

  constructor() {
    this.notificationService = new NotificationService();
  }

  async sendNotification(req: Request, res: Response): Promise<void> {
    try {
      const notification = req.body;
      const result = await this.notificationService.send(notification);
      res.status(200).json(result);
    } catch (error) {
      logger.error('Failed to send notification', { error });
      res.status(500).json({ error: 'Failed to send notification' });
    }
  }

  async sendBulkNotifications(req: Request, res: Response): Promise<void> {
    try {
      const { recipients, channel, templateId, data } = req.body;
      const results = await this.notificationService.sendBulk({
        recipients,
        channel,
        templateId,
        data
      });
      res.status(200).json(results);
    } catch (error) {
      logger.error('Failed to send bulk notifications', { error });
      res.status(500).json({ error: 'Failed to send bulk notifications' });
    }
  }

  async getNotification(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const notification = await this.notificationService.getById(id);
      res.status(200).json(notification);
    } catch (error) {
      logger.error('Failed to get notification', { error });
      res.status(500).json({ error: 'Failed to get notification' });
    }
  }

  async getUserNotifications(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;
      const notifications = await this.notificationService.getByUserId(userId);
      res.status(200).json(notifications);
    } catch (error) {
      logger.error('Failed to get user notifications', { error });
      res.status(500).json({ error: 'Failed to get user notifications' });
    }
  }

  async updateStatus(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const { status } = req.body;
      await this.notificationService.updateStatus(id, status);
      res.status(200).json({ success: true });
    } catch (error) {
      logger.error('Failed to update notification status', { error });
      res.status(500).json({ error: 'Failed to update notification status' });
    }
  }
}
