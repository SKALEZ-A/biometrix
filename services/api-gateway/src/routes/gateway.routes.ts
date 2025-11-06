import { Router } from 'express';
import { authMiddleware } from '../middleware/auth.middleware';
import { rateLimiter } from '../middleware/rate-limiter';
import { circuitBreaker } from '../middleware/circuit-breaker';
import { requestValidator } from '../middleware/request-validator';
import { loggingMiddleware } from '../middleware/logging.middleware';
import { metricsMiddleware } from '../middleware/metrics.middleware';

const router = Router();

// Apply global middleware
router.use(loggingMiddleware);
router.use(metricsMiddleware);
router.use(rateLimiter);

// Health check endpoint
router.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Biometric service routes
router.use('/api/v1/biometric', 
  authMiddleware,
  circuitBreaker('biometric-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.BIOMETRIC_SERVICE_URL || 'http://biometric-service:3001';
    next();
  }
);

// Fraud detection service routes
router.use('/api/v1/fraud-detection',
  authMiddleware,
  circuitBreaker('fraud-detection-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.FRAUD_DETECTION_SERVICE_URL || 'http://fraud-detection-service:3002';
    next();
  }
);

// Transaction service routes
router.use('/api/v1/transactions',
  authMiddleware,
  circuitBreaker('transaction-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.TRANSACTION_SERVICE_URL || 'http://transaction-service:3003';
    next();
  }
);

// User management service routes
router.use('/api/v1/users',
  authMiddleware,
  circuitBreaker('user-management-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.USER_MANAGEMENT_SERVICE_URL || 'http://user-management-service:3004';
    next();
  }
);

// Alert service routes
router.use('/api/v1/alerts',
  authMiddleware,
  circuitBreaker('alert-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.ALERT_SERVICE_URL || 'http://alert-service:3005';
    next();
  }
);

// Compliance service routes
router.use('/api/v1/compliance',
  authMiddleware,
  circuitBreaker('compliance-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.COMPLIANCE_SERVICE_URL || 'http://compliance-service:3006';
    next();
  }
);

// Analytics service routes
router.use('/api/v1/analytics',
  authMiddleware,
  circuitBreaker('analytics-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.ANALYTICS_SERVICE_URL || 'http://analytics-service:3007';
    next();
  }
);

// Merchant protection service routes
router.use('/api/v1/merchant-protection',
  authMiddleware,
  circuitBreaker('merchant-protection-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.MERCHANT_PROTECTION_SERVICE_URL || 'http://merchant-protection-service:3008';
    next();
  }
);

// Voice service routes
router.use('/api/v1/voice',
  authMiddleware,
  circuitBreaker('voice-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.VOICE_SERVICE_URL || 'http://voice-service:3009';
    next();
  }
);

// Webhook service routes
router.use('/api/v1/webhooks',
  authMiddleware,
  circuitBreaker('webhook-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.WEBHOOK_SERVICE_URL || 'http://webhook-service:3010';
    next();
  }
);

// Notification service routes
router.use('/api/v1/notifications',
  authMiddleware,
  circuitBreaker('notification-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.NOTIFICATION_SERVICE_URL || 'http://notification-service:3011';
    next();
  }
);

// Reporting service routes
router.use('/api/v1/reports',
  authMiddleware,
  circuitBreaker('reporting-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.REPORTING_SERVICE_URL || 'http://reporting-service:3012';
    next();
  }
);

// Audit service routes
router.use('/api/v1/audit',
  authMiddleware,
  circuitBreaker('audit-service'),
  requestValidator,
  (req, res, next) => {
    req.serviceUrl = process.env.AUDIT_SERVICE_URL || 'http://audit-service:3013';
    next();
  }
);

export default router;
