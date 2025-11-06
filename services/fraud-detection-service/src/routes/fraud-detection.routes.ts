import { Router } from 'express';
import { FraudDetectionController } from '../controllers/fraud-detection.controller';
import { authMiddleware } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';
import { validationMiddleware } from '../middleware/validation.middleware';

const router = Router();
const fraudDetectionController = new FraudDetectionController();

// Real-time fraud detection
router.post(
  '/detect',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 100, windowMs: 60000 }),
  validationMiddleware.validateFraudDetectionRequest,
  fraudDetectionController.detectFraud.bind(fraudDetectionController)
);

// Batch fraud analysis
router.post(
  '/analyze/batch',
  authMiddleware,
  validationMiddleware.validateBatchAnalysisRequest,
  fraudDetectionController.analyzeBatch.bind(fraudDetectionController)
);

// Risk scoring
router.post(
  '/risk-score',
  authMiddleware,
  rateLimitMiddleware({ maxRequests: 200, windowMs: 60000 }),
  fraudDetectionController.calculateRiskScore.bind(fraudDetectionController)
);

// Pattern analysis
router.get(
  '/patterns',
  authMiddleware,
  fraudDetectionController.getFraudPatterns.bind(fraudDetectionController)
);

router.post(
  '/patterns/analyze',
  authMiddleware,
  fraudDetectionController.analyzePatterns.bind(fraudDetectionController)
);

// Model management
router.get(
  '/models',
  authMiddleware,
  fraudDetectionController.listModels.bind(fraudDetectionController)
);

router.post(
  '/models/:modelId/retrain',
  authMiddleware,
  fraudDetectionController.retrainModel.bind(fraudDetectionController)
);

router.get(
  '/models/:modelId/performance',
  authMiddleware,
  fraudDetectionController.getModelPerformance.bind(fraudDetectionController)
);

// Feature importance
router.get(
  '/features/importance',
  authMiddleware,
  fraudDetectionController.getFeatureImportance.bind(fraudDetectionController)
);

export default router;
