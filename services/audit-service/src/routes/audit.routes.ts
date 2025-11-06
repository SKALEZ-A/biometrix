import { Router } from 'express';
import { AuditController } from '../controllers/audit.controller';
import { authMiddleware } from '../middleware/auth.middleware';

const router = Router();
const auditController = new AuditController();

router.post('/log',
  authMiddleware,
  auditController.createAuditLog.bind(auditController)
);

router.get('/logs',
  authMiddleware,
  auditController.getAuditLogs.bind(auditController)
);

router.get('/logs/:id',
  authMiddleware,
  auditController.getAuditLogById.bind(auditController)
);

router.get('/user/:userId',
  authMiddleware,
  auditController.getUserAuditLogs.bind(auditController)
);

router.get('/entity/:entityType/:entityId',
  authMiddleware,
  auditController.getEntityAuditLogs.bind(auditController)
);

router.get('/search',
  authMiddleware,
  auditController.searchAuditLogs.bind(auditController)
);

router.get('/export',
  authMiddleware,
  auditController.exportAuditLogs.bind(auditController)
);

router.get('/compliance-report',
  authMiddleware,
  auditController.generateComplianceReport.bind(auditController)
);

export default router;
