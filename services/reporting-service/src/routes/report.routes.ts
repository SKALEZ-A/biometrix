import { Router } from 'express';
import { ReportController } from '../controllers/report.controller';
import { ReportGeneratorController } from '../controllers/report-generator.controller';
import { authMiddleware } from '../middleware/auth.middleware';

const router = Router();
const reportController = new ReportController();
const reportGeneratorController = new ReportGeneratorController();

router.post('/generate',
  authMiddleware,
  reportGeneratorController.generateReport.bind(reportGeneratorController)
);

router.get('/',
  authMiddleware,
  reportController.getAllReports.bind(reportController)
);

router.get('/:id',
  authMiddleware,
  reportController.getReportById.bind(reportController)
);

router.get('/:id/download',
  authMiddleware,
  reportController.downloadReport.bind(reportController)
);

router.delete('/:id',
  authMiddleware,
  reportController.deleteReport.bind(reportController)
);

router.post('/schedule',
  authMiddleware,
  reportController.scheduleReport.bind(reportController)
);

router.get('/scheduled/list',
  authMiddleware,
  reportController.getScheduledReports.bind(reportController)
);

export default router;
