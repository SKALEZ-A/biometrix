interface MerchantProfile {
  merchantId: string;
  businessType: string;
  registrationDate: Date;
  totalTransactions: number;
  totalVolume: number;
  chargebackCount: number;
  refundCount: number;
  averageTicketSize: number;
  industryCategory: string;
}

interface RiskAssessment {
  overallRiskScore: number;
  riskCategory: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  factors: {
    chargebackRatio: number;
    refundRatio: number;
    volumeGrowth: number;
    accountAge: number;
    industryRisk: number;
  };
  recommendations: string[];
  limits: {
    dailyTransactionLimit: number;
    monthlyVolumeLimit: number;
    singleTransactionLimit: number;
  };
}

export class MerchantRiskScoringService {
  private industryRiskScores: Map<string, number>;
  private merchantHistoricalData: Map<string, MerchantProfile[]>;

  constructor() {
    this.industryRiskScores = new Map([
      ['e-commerce', 0.6],
      ['travel', 0.7],
      ['digital-goods', 0.8],
      ['gambling', 0.9],
      ['crypto', 0.85],
      ['retail', 0.4],
      ['services', 0.5],
      ['subscription', 0.55]
    ]);
    this.merchantHistoricalData = new Map();
  }

  assessMerchantRisk(profile: MerchantProfile): RiskAssessment {
    const chargebackRatio = profile.totalTransactions > 0
      ? profile.chargebackCount / profile.totalTransactions
      : 0;

    const refundRatio = profile.totalTransactions > 0
      ? profile.refundCount / profile.totalTransactions
      : 0;

    const accountAgeDays = Math.floor(
      (Date.now() - profile.registrationDate.getTime()) / (1000 * 60 * 60 * 24)
    );

    const volumeGrowth = this.calculateVolumeGrowth(profile.merchantId);
    const industryRisk = this.industryRiskScores.get(profile.industryCategory) || 0.5;

    const factors = {
      chargebackRatio,
      refundRatio,
      volumeGrowth,
      accountAge: accountAgeDays,
      industryRisk
    };

    const overallRiskScore = this.calculateOverallRisk(factors);
    const riskCategory = this.categorizeRisk(overallRiskScore);
    const recommendations = this.generateRecommendations(riskCategory, factors);
    const limits = this.calculateLimits(riskCategory, profile);

    return {
      overallRiskScore,
      riskCategory,
      factors,
      recommendations,
      limits
    };
  }

  private calculateOverallRisk(factors: RiskAssessment['factors']): number {
    let score = 0;

    score += factors.chargebackRatio * 100 * 0.3;
    score += factors.refundRatio * 100 * 0.2;
    score += factors.industryRisk * 100 * 0.25;

    if (factors.volumeGrowth > 5) {
      score += 15;
    } else if (factors.volumeGrowth > 3) {
      score += 10;
    }

    if (factors.accountAge < 30) {
      score += 20;
    } else if (factors.accountAge < 90) {
      score += 10;
    }

    return Math.min(score, 100);
  }

  private categorizeRisk(score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    if (score >= 75) return 'CRITICAL';
    if (score >= 55) return 'HIGH';
    if (score >= 35) return 'MEDIUM';
    return 'LOW';
  }

  private generateRecommendations(category: string, factors: RiskAssessment['factors']): string[] {
    const recommendations: string[] = [];

    if (category === 'CRITICAL' || category === 'HIGH') {
      recommendations.push('Implement enhanced transaction monitoring');
      recommendations.push('Require additional verification for high-value transactions');
      recommendations.push('Conduct periodic compliance reviews');
    }

    if (factors.chargebackRatio > 0.01) {
      recommendations.push('Review and improve product descriptions');
      recommendations.push('Enhance customer service response times');
      recommendations.push('Implement clear refund policies');
    }

    if (factors.volumeGrowth > 3) {
      recommendations.push('Monitor for unusual transaction patterns');
      recommendations.push('Verify business legitimacy and operations');
    }

    if (factors.accountAge < 90) {
      recommendations.push('Conduct enhanced due diligence');
      recommendations.push('Implement gradual limit increases');
    }

    return recommendations;
  }

  private calculateLimits(category: string, profile: MerchantProfile): RiskAssessment['limits'] {
    const baseDaily = 50000;
    const baseMonthly = 1000000;
    const baseSingle = 10000;

    const multipliers: Record<string, number> = {
      'LOW': 1.0,
      'MEDIUM': 0.7,
      'HIGH': 0.4,
      'CRITICAL': 0.2
    };

    const multiplier = multipliers[category] || 0.5;

    return {
      dailyTransactionLimit: Math.floor(baseDaily * multiplier),
      monthlyVolumeLimit: Math.floor(baseMonthly * multiplier),
      singleTransactionLimit: Math.floor(baseSingle * multiplier)
    };
  }

  private calculateVolumeGrowth(merchantId: string): number {
    const history = this.merchantHistoricalData.get(merchantId);
    if (!history || history.length < 2) {
      return 0;
    }

    const recent = history[history.length - 1];
    const previous = history[history.length - 2];

    if (previous.totalVolume === 0) {
      return 0;
    }

    return (recent.totalVolume - previous.totalVolume) / previous.totalVolume;
  }

  updateMerchantHistory(profile: MerchantProfile): void {
    const history = this.merchantHistoricalData.get(profile.merchantId) || [];
    history.push(profile);

    if (history.length > 12) {
      history.shift();
    }

    this.merchantHistoricalData.set(profile.merchantId, history);
  }
}
