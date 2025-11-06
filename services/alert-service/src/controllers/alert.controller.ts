import { Request, Response, NextFunction } from 'express';
import { NotificationService } from '../services/notification.service';
import { AlertRoutingService } from '../services/alert-routing.service';

export class AlertController {
  private notificationService: NotificationService;
  private routingService: AlertRoutingService;

  constructor() {
    this.notificationService = new NotificationService();
    this.routingService = new AlertRoutingService();
  }

  async sendAlert(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const alertData = req.body;
      const routes = this.routingService.determineRoutes(alertData);

      const results = await Promise.all(
        routes.map(route => 
          this.notificationService.send(route.channel, route.recipient, alertData)
        )
      );

      res.status(200).json({
        success: true,
        data: {
          alertId: alertData.alertId,
          routesSent: results.length,
          results
        }
      });
    } catch (error) {
      next(error);
    }
  }

  async getAlertHistory(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId, startDate, endDate } = req.query;
      const history = await this.notificationService.getHistory(
        userId as string,
        new Date(startDate as string),
        new Date(endDate as string)
      );

      res.status(200).json({
        success: true,
        data: history
      });
    } catch (error) {
      next(error);
    }
  }
}
