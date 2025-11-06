import twilio from 'twilio';
import { logger } from '../../../packages/shared/src/utils/logger';

export interface SMSOptions {
  to: string;
  message: string;
  from?: string;
}

export class SMSNotificationService {
  private client: twilio.Twilio;
  private fromNumber: string;

  constructor() {
    this.client = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
    this.fromNumber = process.env.TWILIO_PHONE_NUMBER || '';
  }

  async sendSMS(options: SMSOptions): Promise<void> {
    try {
      const message = await this.client.messages.create({
        body: options.message,
        from: options.from || this.fromNumber,
        to: options.to
      });

      logger.info('SMS sent successfully', { messageId: message.sid });
    } catch (error) {
      logger.error('Failed to send SMS', { error, options });
      throw error;
    }
  }

  async sendBulkSMS(recipients: string[], message: string): Promise<void> {
    const promises = recipients.map(to => 
      this.sendSMS({ to, message })
    );

    await Promise.allSettled(promises);
  }
}
