import { Request, Response } from 'express';
import { WebhookSubscriptionService } from '../services/webhook-subscription.service';

export class WebhookSubscriptionController {
  private subscriptionService: WebhookSubscriptionService;

  constructor() {
    this.subscriptionService = new WebhookSubscriptionService();
  }

  async createSubscription(req: Request, res: Response): Promise<void> {
    try {
      const subscription = await this.subscriptionService.createSubscription(req.body);
      res.status(201).json({ success: true, data: subscription });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }

  async getSubscriptions(req: Request, res: Response): Promise<void> {
    try {
      const subscriptions = await this.subscriptionService.getSubscriptions(req.query);
      res.json({ success: true, data: subscriptions });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }

  async updateSubscription(req: Request, res: Response): Promise<void> {
    try {
      const subscription = await this.subscriptionService.updateSubscription(
        req.params.id,
        req.body
      );
      res.json({ success: true, data: subscription });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }

  async deleteSubscription(req: Request, res: Response): Promise<void> {
    try {
      await this.subscriptionService.deleteSubscription(req.params.id);
      res.json({ success: true, message: 'Subscription deleted' });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }

  async testWebhook(req: Request, res: Response): Promise<void> {
    try {
      const result = await this.subscriptionService.testWebhook(req.params.id);
      res.json({ success: true, data: result });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }
}
