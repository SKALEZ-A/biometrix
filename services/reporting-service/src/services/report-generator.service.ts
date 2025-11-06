import { ReportService } from './report.service';
import PDFDocument from 'pdfkit';
import ExcelJS from 'exceljs';
import { createObjectCsvStringifier } from 'csv-writer';

export interface ReportData {
  title: string;
  generatedAt: Date;
  data: any[];
  metadata?: Record<string, any>;
}

export class ReportGeneratorService {
  private reportService: ReportService;

  constructor() {
    this.reportService = new ReportService();
  }

  async generatePDFReport(reportData: ReportData): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const doc = new PDFDocument();
      const chunks: Buffer[] = [];

      doc.on('data', (chunk) => chunks.push(chunk));
      doc.on('end', () => resolve(Buffer.concat(chunks)));
      doc.on('error', reject);

      doc.fontSize(20).text(reportData.title, { align: 'center' });
      doc.moveDown();
      doc.fontSize(12).text(`Generated: ${reportData.generatedAt.toISOString()}`);
      doc.moveDown();

      reportData.data.forEach((item, index) => {
        doc.fontSize(10).text(`${index + 1}. ${JSON.stringify(item)}`);
        doc.moveDown(0.5);
      });

      doc.end();
    });
  }

  async generateExcelReport(reportData: ReportData): Promise<Buffer> {
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Report');

    worksheet.addRow([reportData.title]);
    worksheet.addRow([`Generated: ${reportData.generatedAt.toISOString()}`]);
    worksheet.addRow([]);

    if (reportData.data.length > 0) {
      const headers = Object.keys(reportData.data[0]);
      worksheet.addRow(headers);

      reportData.data.forEach(item => {
        const row = headers.map(header => item[header]);
        worksheet.addRow(row);
      });
    }

    return await workbook.xlsx.writeBuffer() as Buffer;
  }

  async generateCSVReport(reportData: ReportData): Promise<string> {
    if (reportData.data.length === 0) {
      return '';
    }

    const headers = Object.keys(reportData.data[0]);
    const csvStringifier = createObjectCsvStringifier({
      header: headers.map(h => ({ id: h, title: h }))
    });

    const headerString = csvStringifier.getHeaderString();
    const recordsString = csvStringifier.stringifyRecords(reportData.data);

    return `${reportData.title}\nGenerated: ${reportData.generatedAt.toISOString()}\n\n${headerString}${recordsString}`;
  }

  async generateJSONReport(reportData: ReportData): Promise<string> {
    return JSON.stringify(reportData, null, 2);
  }
}
