import { Request, Response, NextFunction } from 'express';
import { RegulatoryReportingService } from '../services/regulatory-reporting.service';
import { AMLScreeningService } from '../services/aml-screening.service';
import { KYCVerificationService } from '../services/kyc-verification.service';

export class ComplianceController {
  private reportingService: RegulatoryReportingService;
  private amlService: AMLScreeningService;
  private kycService: KYCVerificationService;

  constructor() {
    this.reportingService = new RegulatoryReportingService();
    this.amlService = new AMLScreeningService();
    this.kycService = new KYCVerificationService();
  }

  async generateReport(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { reportType, startDate, endDate } = req.body;
      const report = await this.reportingService.generateReport(reportType, startDate, endDate);

      res.status(200).json({
        success: true,
        data: report
      });
    } catch (error) {
      next(error);
    }
  }

  async screenTransaction(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const transactionData = req.body;
      const screeningResult = await this.amlService.screenTransaction(transactionData);

      res.status(200).json({
        success: true,
        data: screeningResult
      });
    } catch (error) {
      next(error);
    }
  }

  async verifyCustomer(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const customerData = req.body;
      const verificationResult = await this.kycService.verifyCustomer(customerData);

      res.status(200).json({
        success: true,
        data: verificationResult
      });
    } catch (error) {
      next(error);
    }
  }
}
