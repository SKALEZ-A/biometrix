import { Router } from 'express';
import { AnalyticsController } from '../controllers/analytics.controller';
import { authMiddleware } from '../middleware/auth.middleware';

const router = Router();
const analyticsController = new AnalyticsController();

router.get('/dashboard',
  authMiddleware,
  analyticsController.getDashboardMetrics.bind(analyticsController)
);

router.get('/fraud-trends',
  authMiddleware,
  analyticsController.getFraudTrends.bind(analyticsController)
);

router.get('/transaction-volume',
  authMiddleware,
  analyticsController.getTransactionVolume.bind(analyticsController)
);

router.get('/risk-distribution',
  authMiddleware,
  analyticsController.getRiskDistribution.bind(analyticsController)
);

router.get('/detection-accuracy',
  authMiddleware,
  analyticsController.getDetectionAccuracy.bind(analyticsController)
);

router.get('/geographic-analysis',
  authMiddleware,
  analyticsController.getGeographicAnalysis.bind(analyticsController)
);

router.get('/time-series',
  authMiddleware,
  analyticsController.getTimeSeriesData.bind(analyticsController)
);

router.get('/top-merchants',
  authMiddleware,
  analyticsController.getTopMerchants.bind(analyticsController)
);

router.get('/alert-statistics',
  authMiddleware,
  analyticsController.getAlertStatistics.bind(analyticsController)
);

router.post('/custom-query',
  authMiddleware,
  analyticsController.executeCustomQuery.bind(analyticsController)
);

export default router;
