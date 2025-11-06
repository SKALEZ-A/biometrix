import { EventEmitter } from 'events';

interface ChargebackRiskFactors {
  transactionId: string;
  merchantId: string;
  customerId: string;
  amount: number;
  productCategory: string;
  deliveryStatus: string;
  customerHistory: {
    totalTransactions: number;
    chargebackCount: number;
    accountAge: number;
  };
  transactionDetails: {
    isFirstPurchase: boolean;
    isHighValue: boolean;
    isInternational: boolean;
    hasTracking: boolean;
  };
}

interface ChargebackPrediction {
  riskScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  preventionActions: string[];
  estimatedLoss: number;
  confidence: number;
}

export class ChargebackPreventionService extends EventEmitter {
  private chargebackHistory: Map<string, number>;
  private merchantRiskProfiles: Map<string, { chargebackRate: number; avgAmount: number }>;

  constructor() {
    super();
    this.chargebackHistory = new Map();
    this.merchantRiskProfiles = new Map();
  }

  async assessChargebackRisk(factors: ChargebackRiskFactors): Promise<ChargebackPrediction> {
    const riskScore = this.calculateRiskScore(factors);
    const riskLevel = this.determineRiskLevel(riskScore);
    const preventionActions = this.generatePreventionActions(riskLevel, factors);
    const estimatedLoss = this.estimatePotentialLoss(factors.amount, riskScore);
    const confidence = this.calculateConfidence(factors);

    if (riskLevel === 'HIGH' || riskLevel === 'CRITICAL') {
      this.emit('high-chargeback-risk', {
        transactionId: factors.transactionId,
        merchantId: factors.merchantId,
        riskScore,
        timestamp: new Date()
      });
    }

    return {
      riskScore,
      riskLevel,
      preventionActions,
      estimatedLoss,
      confidence
    };
  }

  private calculateRiskScore(factors: ChargebackRiskFactors): number {
    let score = 0;

    const customerChargebackRate = factors.customerHistory.totalTransactions > 0
      ? factors.customerHistory.chargebackCount / factors.customerHistory.totalTransactions
      : 0;
    score += customerChargebackRate * 30;

    if (factors.transactionDetails.isFirstPurchase) {
      score += 15;
    }

    if (factors.transactionDetails.isHighValue) {
      score += 20;
    }

    if (factors.transactionDetails.isInternational) {
      score += 10;
    }

    if (!factors.transactionDetails.hasTracking) {
      score += 15;
    }

    if (factors.customerHistory.accountAge < 30) {
      score += 10;
    }

    const merchantProfile = this.merchantRiskProfiles.get(factors.merchantId);
    if (merchantProfile && merchantProfile.chargebackRate > 0.02) {
      score += 20;
    }

    return Math.min(score, 100);
  }

  private determineRiskLevel(score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    if (score >= 80) return 'CRITICAL';
    if (score >= 60) return 'HIGH';
    if (score >= 40) return 'MEDIUM';
    return 'LOW';
  }

  private generatePreventionActions(riskLevel: string, factors: ChargebackRiskFactors): string[] {
    const actions: string[] = [];

    if (riskLevel === 'CRITICAL') {
      actions.push('Hold transaction for manual review');
      actions.push('Contact customer for verification');
      actions.push('Require additional authentication');
    }

    if (riskLevel === 'HIGH' || riskLevel === 'CRITICAL') {
      actions.push('Enable delivery confirmation');
      actions.push('Require signature on delivery');
      actions.push('Send proactive customer communication');
    }

    if (!factors.transactionDetails.hasTracking) {
      actions.push('Add tracking to shipment');
    }

    if (factors.transactionDetails.isHighValue) {
      actions.push('Purchase shipping insurance');
      actions.push('Use expedited shipping with tracking');
    }

    actions.push('Document all customer interactions');
    actions.push('Save transaction evidence');

    return actions;
  }

  private estimatePotentialLoss(amount: number, riskScore: number): number {
    const chargebackFee = 25;
    const probabilityOfChargeback = riskScore / 100;
    return (amount + chargebackFee) * probabilityOfChargeback;
  }

  private calculateConfidence(factors: ChargebackRiskFactors): number {
    let confidence = 0.5;

    if (factors.customerHistory.totalTransactions > 10) {
      confidence += 0.2;
    }

    if (factors.customerHistory.accountAge > 90) {
      confidence += 0.15;
    }

    if (factors.transactionDetails.hasTracking) {
      confidence += 0.1;
    }

    return Math.min(confidence, 1.0);
  }

  recordChargeback(transactionId: string, merchantId: string): void {
    const currentCount = this.chargebackHistory.get(merchantId) || 0;
    this.chargebackHistory.set(merchantId, currentCount + 1);

    const profile = this.merchantRiskProfiles.get(merchantId);
    if (profile) {
      profile.chargebackRate = (profile.chargebackRate * 0.9) + 0.1;
    }
  }

  updateMerchantProfile(merchantId: string, chargebackRate: number, avgAmount: number): void {
    this.merchantRiskProfiles.set(merchantId, { chargebackRate, avgAmount });
  }
}
