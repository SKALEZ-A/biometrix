import { Router } from 'express';
import { MerchantProtectionController } from '../controllers/merchant-protection.controller';
import { authMiddleware } from '../middleware/auth.middleware';
import { validationMiddleware } from '../middleware/validation.middleware';

const router = Router();
const merchantProtectionController = new MerchantProtectionController();

// Merchant risk assessment
router.post(
  '/merchants/:merchantId/risk-assessment',
  authMiddleware,
  validationMiddleware.validateMerchantId,
  merchantProtectionController.assessMerchantRisk.bind(merchantProtectionController)
);

router.get(
  '/merchants/:merchantId/risk-score',
  authMiddleware,
  merchantProtectionController.getMerchantRiskScore.bind(merchantProtectionController)
);

// Chargeback prevention
router.post(
  '/chargebacks/predict',
  authMiddleware,
  merchantProtectionController.predictChargeback.bind(merchantProtectionController)
);

router.get(
  '/chargebacks/analytics',
  authMiddleware,
  merchantProtectionController.getChargebackAnalytics.bind(merchantProtectionController)
);

router.post(
  '/chargebacks/:chargebackId/dispute',
  authMiddleware,
  merchantProtectionController.initiateDispute.bind(merchantProtectionController)
);

// Dispute resolution
router.get(
  '/disputes',
  authMiddleware,
  merchantProtectionController.listDisputes.bind(merchantProtectionController)
);

router.get(
  '/disputes/:disputeId',
  authMiddleware,
  merchantProtectionController.getDisputeDetails.bind(merchantProtectionController)
);

router.post(
  '/disputes/:disputeId/evidence',
  authMiddleware,
  merchantProtectionController.submitEvidence.bind(merchantProtectionController)
);

router.patch(
  '/disputes/:disputeId/status',
  authMiddleware,
  merchantProtectionController.updateDisputeStatus.bind(merchantProtectionController)
);

// Merchant monitoring
router.get(
  '/merchants/:merchantId/transactions/suspicious',
  authMiddleware,
  merchantProtectionController.getSuspiciousTransactions.bind(merchantProtectionController)
);

router.post(
  '/merchants/:merchantId/alerts/configure',
  authMiddleware,
  merchantProtectionController.configureAlerts.bind(merchantProtectionController)
);

export default router;
