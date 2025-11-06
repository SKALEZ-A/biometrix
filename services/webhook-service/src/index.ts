import express, { Application } from 'express';
import { webhookRoutes } from './routes/webhook.routes';
import { subscriptionRoutes } from './routes/subscription.routes';
import { errorHandler } from './middleware/error-handler.middleware';
import { authMiddleware } from './middleware/auth.middleware';
import { rateLimiter } from './middleware/rate-limiter.middleware';
import { webhookValidator } from './middleware/webhook-validator.middleware';
import { config } from './config/app.config';
import { logger } from '@shared/utils/logger';
import { connectDatabase } from './config/database.config';
import { WebhookProcessor } from './services/webhook-processor.service';
import { RetryService } from './services/retry.service';

const app: Application = express();

app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
app.use(rateLimiter);

// Public webhook endpoint (no auth required)
app.use('/webhooks/receive', webhookValidator, webhookRoutes);

// Protected subscription management endpoints
app.use('/api/subscriptions', authMiddleware, subscriptionRoutes);

// Health check
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    service: 'webhook-service',
    timestamp: new Date().toISOString(),
  });
});

app.use(errorHandler);

const startServer = async () => {
  try {
    await connectDatabase();
    
    // Initialize webhook processor
    const webhookProcessor = new WebhookProcessor();
    await webhookProcessor.start();
    
    // Initialize retry service
    const retryService = new RetryService();
    await retryService.start();
    
    const PORT = config.port || 4013;
    app.listen(PORT, () => {
      logger.info(`Webhook Service running on port ${PORT}`);
    });
  } catch (error) {
    logger.error('Failed to start Webhook Service', { error });
    process.exit(1);
  }
};

startServer();

export default app;
