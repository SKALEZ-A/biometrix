import express from 'express';
import { ReportController } from './controllers/report.controller';
import { authMiddleware } from './middleware/auth.middleware';
import { logger } from '@shared/utils/logger';

const app = express();
const port = process.env.PORT || 3010;

app.use(express.json());

const reportController = new ReportController();

app.post('/reports/generate', authMiddleware, reportController.generateReport.bind(reportController));
app.get('/reports/:id', authMiddleware, reportController.getReport.bind(reportController));
app.get('/reports', authMiddleware, reportController.listReports.bind(reportController));
app.post('/reports/:id/export', authMiddleware, reportController.exportReport.bind(reportController));
app.get('/reports/scheduled', authMiddleware, reportController.getScheduledReports.bind(reportController));

app.listen(port, () => {
  logger.info(`Reporting service listening on port ${port}`);
});
