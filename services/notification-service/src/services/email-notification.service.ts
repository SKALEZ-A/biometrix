import nodemailer from 'nodemailer';
import { logger } from '../../../packages/shared/src/utils/logger';

export interface EmailOptions {
  to: string | string[];
  subject: string;
  html?: string;
  text?: string;
  attachments?: any[];
}

export class EmailNotificationService {
  private transporter: nodemailer.Transporter;

  constructor() {
    this.transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: process.env.SMTP_SECURE === 'true',
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
      }
    });
  }

  async sendEmail(options: EmailOptions): Promise<void> {
    try {
      const info = await this.transporter.sendMail({
        from: process.env.SMTP_FROM || 'noreply@fraudprevention.com',
        to: Array.isArray(options.to) ? options.to.join(', ') : options.to,
        subject: options.subject,
        text: options.text,
        html: options.html,
        attachments: options.attachments
      });

      logger.info('Email sent successfully', { messageId: info.messageId });
    } catch (error) {
      logger.error('Failed to send email', { error, options });
      throw error;
    }
  }

  async sendTemplateEmail(template: string, data: any, to: string | string[]): Promise<void> {
    const templates = {
      'fraud-alert': {
        subject: 'Fraud Alert - Suspicious Activity Detected',
        html: `<h1>Fraud Alert</h1><p>Suspicious activity detected: ${data.description}</p>`
      },
      'transaction-blocked': {
        subject: 'Transaction Blocked',
        html: `<h1>Transaction Blocked</h1><p>Transaction ${data.transactionId} was blocked due to fraud detection.</p>`
      },
      'welcome': {
        subject: 'Welcome to Fraud Prevention System',
        html: `<h1>Welcome ${data.name}!</h1><p>Your account has been created successfully.</p>`
      }
    };

    const templateData = templates[template];
    if (!templateData) {
      throw new Error(`Template ${template} not found`);
    }

    await this.sendEmail({
      to,
      subject: templateData.subject,
      html: templateData.html
    });
  }
}
