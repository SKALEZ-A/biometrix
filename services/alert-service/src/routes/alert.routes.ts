import { Router } from 'express';
import { AlertController } from '../controllers/alert.controller';
import { authMiddleware } from '../middleware/auth.middleware';
import { validationMiddleware } from '../middleware/validation.middleware';
import { alertValidators } from '../validators/alert.validators';

const router = Router();
const alertController = new AlertController();

router.post(
  '/alerts',
  authMiddleware,
  validationMiddleware(alertValidators.createAlert),
  alertController.createAlert.bind(alertController)
);

router.get(
  '/alerts',
  authMiddleware,
  validationMiddleware(alertValidators.getAlerts),
  alertController.getAlerts.bind(alertController)
);

router.get(
  '/alerts/:id',
  authMiddleware,
  validationMiddleware(alertValidators.getAlertById),
  alertController.getAlertById.bind(alertController)
);

router.patch(
  '/alerts/:id',
  authMiddleware,
  validationMiddleware(alertValidators.updateAlert),
  alertController.updateAlert.bind(alertController)
);

router.post(
  '/alerts/:id/acknowledge',
  authMiddleware,
  validationMiddleware(alertValidators.acknowledgeAlert),
  alertController.acknowledgeAlert.bind(alertController)
);

router.post(
  '/alerts/:id/resolve',
  authMiddleware,
  validationMiddleware(alertValidators.resolveAlert),
  alertController.resolveAlert.bind(alertController)
);

router.post(
  '/alerts/:id/escalate',
  authMiddleware,
  validationMiddleware(alertValidators.escalateAlert),
  alertController.escalateAlert.bind(alertController)
);

router.get(
  '/alerts/stats/summary',
  authMiddleware,
  alertController.getAlertStats.bind(alertController)
);

router.post(
  '/alerts/bulk/acknowledge',
  authMiddleware,
  validationMiddleware(alertValidators.bulkAcknowledge),
  alertController.bulkAcknowledge.bind(alertController)
);

export default router;
