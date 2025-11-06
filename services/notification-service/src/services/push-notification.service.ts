import admin from 'firebase-admin';
import { logger } from '../../../packages/shared/src/utils/logger';

export interface PushNotificationOptions {
  token: string;
  title: string;
  body: string;
  data?: Record<string, string>;
}

export class PushNotificationService {
  private messaging: admin.messaging.Messaging;

  constructor() {
    if (!admin.apps.length) {
      admin.initializeApp({
        credential: admin.credential.cert({
          projectId: process.env.FIREBASE_PROJECT_ID,
          clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
          privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n')
        })
      });
    }
    this.messaging = admin.messaging();
  }

  async sendPushNotification(options: PushNotificationOptions): Promise<void> {
    try {
      const message = await this.messaging.send({
        token: options.token,
        notification: {
          title: options.title,
          body: options.body
        },
        data: options.data
      });

      logger.info('Push notification sent', { messageId: message });
    } catch (error) {
      logger.error('Failed to send push notification', { error, options });
      throw error;
    }
  }

  async sendMulticast(tokens: string[], title: string, body: string): Promise<void> {
    try {
      const response = await this.messaging.sendMulticast({
        tokens,
        notification: { title, body }
      });

      logger.info('Multicast notification sent', {
        successCount: response.successCount,
        failureCount: response.failureCount
      });
    } catch (error) {
      logger.error('Failed to send multicast notification', { error });
      throw error;
    }
  }
}
