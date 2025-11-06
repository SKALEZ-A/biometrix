import { Request, Response, NextFunction } from 'express';
import { ChargebackPreventionService } from '../services/chargeback-prevention.service';
import { MerchantRiskScoringService } from '../services/merchant-risk-scoring.service';

export class MerchantProtectionController {
  private chargebackService: ChargebackPreventionService;
  private riskScoringService: MerchantRiskScoringService;

  constructor() {
    this.chargebackService = new ChargebackPreventionService();
    this.riskScoringService = new MerchantRiskScoringService();
  }

  async assessChargebackRisk(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const riskFactors = req.body;
      const prediction = await this.chargebackService.assessChargebackRisk(riskFactors);

      res.status(200).json({
        success: true,
        data: prediction
      });
    } catch (error) {
      next(error);
    }
  }

  async assessMerchantRisk(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const merchantProfile = req.body;
      const assessment = this.riskScoringService.assessMerchantRisk(merchantProfile);

      res.status(200).json({
        success: true,
        data: assessment
      });
    } catch (error) {
      next(error);
    }
  }

  async recordChargeback(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { transactionId, merchantId } = req.body;
      this.chargebackService.recordChargeback(transactionId, merchantId);

      res.status(200).json({
        success: true,
        message: 'Chargeback recorded successfully'
      });
    } catch (error) {
      next(error);
    }
  }
}
