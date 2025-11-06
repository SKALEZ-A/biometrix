import { Request, Response } from 'express';
import { webhookService } from '../services/webhook.service';
import { logger } from '../../../packages/shared/src/utils/logger';

export class WebhookController {
  public async registerWebhook(req: Request, res: Response): Promise<void> {
    try {
      const { url, events, secret } = req.body;

      if (!url || !events || !Array.isArray(events)) {
        res.status(400).json({
          error: 'Bad Request',
          message: 'url and events array are required'
        });
        return;
      }

      const webhook = await webhookService.registerWebhook({
        url,
        events,
        secret,
        active: true
      });

      res.status(201).json({
        success: true,
        data: webhook
      });
    } catch (error) {
      logger.error('Error registering webhook:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to register webhook'
      });
    }
  }

  public async getWebhooks(req: Request, res: Response): Promise<void> {
    try {
      const webhooks = await webhookService.getAllWebhooks();

      res.status(200).json({
        success: true,
        data: webhooks
      });
    } catch (error) {
      logger.error('Error getting webhooks:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to retrieve webhooks'
      });
    }
  }

  public async getWebhook(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const webhook = await webhookService.getWebhook(id);

      if (!webhook) {
        res.status(404).json({
          error: 'Not Found',
          message: 'Webhook not found'
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: webhook
      });
    } catch (error) {
      logger.error('Error getting webhook:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to retrieve webhook'
      });
    }
  }

  public async updateWebhook(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const updates = req.body;

      const webhook = await webhookService.updateWebhook(id, updates);

      if (!webhook) {
        res.status(404).json({
          error: 'Not Found',
          message: 'Webhook not found'
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: webhook
      });
    } catch (error) {
      logger.error('Error updating webhook:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to update webhook'
      });
    }
  }

  public async deleteWebhook(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const deleted = await webhookService.deleteWebhook(id);

      if (!deleted) {
        res.status(404).json({
          error: 'Not Found',
          message: 'Webhook not found'
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: 'Webhook deleted successfully'
      });
    } catch (error) {
      logger.error('Error deleting webhook:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to delete webhook'
      });
    }
  }

  public async testWebhook(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const result = await webhookService.testWebhook(id);

      res.status(200).json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error testing webhook:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to test webhook'
      });
    }
  }
}

export const webhookController = new WebhookController();
