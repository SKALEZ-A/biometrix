import { Request, Response } from 'express';
import { ReportGeneratorService } from '../services/report-generator.service';

export class ReportGeneratorController {
  private generatorService: ReportGeneratorService;

  constructor() {
    this.generatorService = new ReportGeneratorService();
  }

  async generateReport(req: Request, res: Response): Promise<void> {
    try {
      const { format, title, data } = req.body;
      const reportData = {
        title,
        generatedAt: new Date(),
        data
      };

      let result: Buffer | string;
      let contentType: string;
      let filename: string;

      switch (format) {
        case 'pdf':
          result = await this.generatorService.generatePDFReport(reportData);
          contentType = 'application/pdf';
          filename = `${title}.pdf`;
          break;
        case 'excel':
          result = await this.generatorService.generateExcelReport(reportData);
          contentType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
          filename = `${title}.xlsx`;
          break;
        case 'csv':
          result = await this.generatorService.generateCSVReport(reportData);
          contentType = 'text/csv';
          filename = `${title}.csv`;
          break;
        case 'json':
          result = await this.generatorService.generateJSONReport(reportData);
          contentType = 'application/json';
          filename = `${title}.json`;
          break;
        default:
          res.status(400).json({ success: false, error: 'Invalid format' });
          return;
      }

      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.send(result);
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }
}
