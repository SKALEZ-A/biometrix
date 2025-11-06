import { Router } from 'express';
import { ComplianceController } from '../controllers/compliance.controller';
import { authMiddleware } from '../middleware/auth.middleware';

const router = Router();
const complianceController = new ComplianceController();

router.post(
  '/kyc/verify',
  authMiddleware,
  complianceController.verifyKYC.bind(complianceController)
);

router.post(
  '/aml/screen',
  authMiddleware,
  complianceController.screenAML.bind(complianceController)
);

router.post(
  '/sanctions/check',
  authMiddleware,
  complianceController.checkSanctions.bind(complianceController)
);

router.get(
  '/reports/:id',
  authMiddleware,
  complianceController.getReport.bind(complianceController)
);

router.post(
  '/reports/generate',
  authMiddleware,
  complianceController.generateReport.bind(complianceController)
);

router.get(
  '/compliance/status/:userId',
  authMiddleware,
  complianceController.getComplianceStatus.bind(complianceController)
);

export default router;
