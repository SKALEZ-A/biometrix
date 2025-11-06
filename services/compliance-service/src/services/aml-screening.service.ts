import { SanctionsList, PEPList, WatchlistMatch, RiskRating } from '@shared/types/compliance.types';
import { Logger } from '@shared/utils/logger';
import { MetricsCollector } from '@shared/utils/metrics';

export class AMLScreeningService {
  private readonly logger = new Logger('AMLScreeningService');
  private readonly metrics = new MetricsCollector();
  private sanctionsLists: Map<string, SanctionsList> = new Map();
  private pepLists: Map<string, PEPList> = new Map();
  private watchlists: Map<string, any[]> = new Map();

  constructor() {
    this.initializeScreeningLists();
  }

  private initializeScreeningLists(): void {
    this.sanctionsLists.set('OFAC_SDN', {
      id: 'OFAC_SDN',
      name: 'OFAC Specially Designated Nationals',
      source: 'US Treasury',
      lastUpdated: Date.now(),
      entries: []
    });

    this.sanctionsLists.set('UN_SANCTIONS', {
      id: 'UN_SANCTIONS',
      name: 'UN Security Council Sanctions List',
      source: 'United Nations',
      lastUpdated: Date.now(),
      entries: []
    });

    this.sanctionsLists.set('EU_SANCTIONS', {
      id: 'EU_SANCTIONS',
      name: 'EU Consolidated Sanctions List',
      source: 'European Union',
      lastUpdated: Date.now(),
      entries: []
    });

    this.pepLists.set('WORLD_PEP', {
      id: 'WORLD_PEP',
      name: 'Global Politically Exposed Persons',
      source: 'Multiple Sources',
      lastUpdated: Date.now(),
      entries: []
    });
  }

  async screenEntity(
    name: string,
    dateOfBirth?: string,
    nationality?: string,
    identificationNumber?: string
  ): Promise<WatchlistMatch[]> {
    const startTime = Date.now();
    const matches: WatchlistMatch[] = [];

    try {
      const sanctionMatches = await this.screenAgainstSanctions(name, dateOfBirth, nationality);
      matches.push(...sanctionMatches);

      const pepMatches = await this.screenAgainstPEP(name, dateOfBirth, nationality);
      matches.push(...pepMatches);

      const adverseMediaMatches = await this.screenAdverseMedia(name);
      matches.push(...adverseMediaMatches);

      this.metrics.recordLatency('aml_screening', Date.now() - startTime);
      this.metrics.incrementCounter('aml_screenings', { 
        result: matches.length > 0 ? 'match' : 'no_match' 
      });

      return matches;
    } catch (error) {
      this.logger.error('AML screening failed', error);
      throw new Error('Failed to complete AML screening');
    }
  }

  private async screenAgainstSanctions(
    name: string,
    dateOfBirth?: string,
    nationality?: string
  ): Promise<WatchlistMatch[]> {
    const matches: WatchlistMatch[] = [];

    for (const [listId, sanctionsList] of this.sanctionsLists.entries()) {
      const listMatches = this.fuzzyMatchAgainstList(
        name,
        sanctionsList.entries,
        dateOfBirth,
        nationality
      );

      for (const match of listMatches) {
        matches.push({
          listId,
          listName: sanctionsList.name,
          matchType: 'SANCTION',
          matchedName: match.name,
          matchScore: match.score,
          riskRating: this.calculateRiskRating(match.score, 'SANCTION'),
          additionalInfo: match.additionalInfo,
          timestamp: Date.now()
        });
      }
    }

    return matches;
  }

  private async screenAgainstPEP(
    name: string,
    dateOfBirth?: string,
    nationality?: string
  ): Promise<WatchlistMatch[]> {
    const matches: WatchlistMatch[] = [];

    for (const [listId, pepList] of this.pepLists.entries()) {
      const listMatches = this.fuzzyMatchAgainstList(
        name,
        pepList.entries,
        dateOfBirth,
        nationality
      );

      for (const match of listMatches) {
        matches.push({
          listId,
          listName: pepList.name,
          matchType: 'PEP',
          matchedName: match.name,
          matchScore: match.score,
          riskRating: this.calculateRiskRating(match.score, 'PEP'),
          additionalInfo: match.additionalInfo,
          timestamp: Date.now()
        });
      }
    }

    return matches;
  }

  private async screenAdverseMedia(name: string): Promise<WatchlistMatch[]> {
    const matches: WatchlistMatch[] = [];
    
    const adverseMediaKeywords = [
      'fraud', 'money laundering', 'corruption', 'bribery',
      'embezzlement', 'terrorist financing', 'sanctions violation',
      'financial crime', 'criminal investigation', 'indicted'
    ];

    const searchResults = await this.searchAdverseMedia(name, adverseMediaKeywords);

    for (const result of searchResults) {
      matches.push({
        listId: 'ADVERSE_MEDIA',
        listName: 'Adverse Media Screening',
        matchType: 'ADVERSE_MEDIA',
        matchedName: name,
        matchScore: result.relevanceScore,
        riskRating: this.calculateRiskRating(result.relevanceScore, 'ADVERSE_MEDIA'),
        additionalInfo: {
          source: result.source,
          headline: result.headline,
          date: result.date,
          keywords: result.matchedKeywords
        },
        timestamp: Date.now()
      });
    }

    return matches;
  }

  private fuzzyMatchAgainstList(
    searchName: string,
    entries: any[],
    dateOfBirth?: string,
    nationality?: string
  ): any[] {
    const matches = [];
    const normalizedSearch = this.normalizeName(searchName);

    for (const entry of entries) {
      const normalizedEntry = this.normalizeName(entry.name);
      const nameScore = this.calculateNameSimilarity(normalizedSearch, normalizedEntry);

      if (nameScore >= 0.75) {
        let totalScore = nameScore;
        let matchCount = 1;

        if (dateOfBirth && entry.dateOfBirth) {
          const dobMatch = this.compareDates(dateOfBirth, entry.dateOfBirth);
          totalScore += dobMatch;
          matchCount++;
        }

        if (nationality && entry.nationality) {
          const nationalityMatch = nationality.toLowerCase() === entry.nationality.toLowerCase() ? 1.0 : 0.0;
          totalScore += nationalityMatch;
          matchCount++;
        }

        const finalScore = totalScore / matchCount;

        if (finalScore >= 0.70) {
          matches.push({
            name: entry.name,
            score: finalScore,
            additionalInfo: {
              dateOfBirth: entry.dateOfBirth,
              nationality: entry.nationality,
              aliases: entry.aliases || [],
              program: entry.program,
              listingDate: entry.listingDate
            }
          });
        }
      }
    }

    return matches.sort((a, b) => b.score - a.score);
  }

  private normalizeName(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private calculateNameSimilarity(name1: string, name2: string): number {
    const distance = this.levenshteinDistance(name1, name2);
    const maxLength = Math.max(name1.length, name2.length);
    
    if (maxLength === 0) return 1.0;
    
    const similarity = 1 - (distance / maxLength);
    
    const tokens1 = name1.split(' ');
    const tokens2 = name2.split(' ');
    const tokenOverlap = this.calculateTokenOverlap(tokens1, tokens2);
    
    return (similarity * 0.6) + (tokenOverlap * 0.4);
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix: number[][] = [];

    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[str2.length][str1.length];
  }

  private calculateTokenOverlap(tokens1: string[], tokens2: string[]): number {
    const set1 = new Set(tokens1);
    const set2 = new Set(tokens2);
    
    let overlap = 0;
    for (const token of set1) {
      if (set2.has(token)) {
        overlap++;
      }
    }

    const maxTokens = Math.max(tokens1.length, tokens2.length);
    return maxTokens > 0 ? overlap / maxTokens : 0;
  }

  private compareDates(date1: string, date2: string): number {
    try {
      const d1 = new Date(date1);
      const d2 = new Date(date2);
      
      if (d1.getTime() === d2.getTime()) {
        return 1.0;
      }
      
      const daysDiff = Math.abs((d1.getTime() - d2.getTime()) / (1000 * 60 * 60 * 24));
      
      if (daysDiff <= 1) return 0.9;
      if (daysDiff <= 7) return 0.7;
      if (daysDiff <= 30) return 0.5;
      if (daysDiff <= 365) return 0.3;
      
      return 0.0;
    } catch (error) {
      return 0.0;
    }
  }

  private async searchAdverseMedia(name: string, keywords: string[]): Promise<any[]> {
    const results = [];
    
    const mockArticles = [
      {
        source: 'Financial Times',
        headline: `Investigation into ${name} financial dealings`,
        date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        content: 'fraud investigation money laundering',
        relevance: 0.85
      },
      {
        source: 'Reuters',
        headline: `${name} faces corruption charges`,
        date: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(),
        content: 'corruption bribery criminal charges',
        relevance: 0.78
      }
    ];

    for (const article of mockArticles) {
      const matchedKeywords = keywords.filter(keyword => 
        article.content.toLowerCase().includes(keyword.toLowerCase())
      );

      if (matchedKeywords.length > 0) {
        results.push({
          source: article.source,
          headline: article.headline,
          date: article.date,
          relevanceScore: article.relevance,
          matchedKeywords
        });
      }
    }

    return results;
  }

  private calculateRiskRating(matchScore: number, matchType: string): RiskRating {
    const typeWeights = {
      'SANCTION': 1.0,
      'PEP': 0.7,
      'ADVERSE_MEDIA': 0.5
    };

    const weight = typeWeights[matchType] || 0.5;
    const adjustedScore = matchScore * weight;

    if (adjustedScore >= 0.9) return 'CRITICAL';
    if (adjustedScore >= 0.75) return 'HIGH';
    if (adjustedScore >= 0.60) return 'MEDIUM';
    return 'LOW';
  }

  async performEnhancedDueDiligence(
    entityId: string,
    entityName: string,
    entityType: 'INDIVIDUAL' | 'ORGANIZATION'
  ): Promise<any> {
    const screeningResults = await this.screenEntity(entityName);
    
    const ownershipStructure = await this.investigateOwnership(entityId);
    const transactionPatterns = await this.analyzeTransactionPatterns(entityId);
    const geographicRisk = await this.assessGeographicRisk(entityId);
    const industryRisk = await this.assessIndustryRisk(entityId);

    const overallRisk = this.calculateOverallRisk({
      screeningMatches: screeningResults.length,
      ownershipComplexity: ownershipStructure.complexity,
      transactionRisk: transactionPatterns.riskScore,
      geographicRisk: geographicRisk.score,
      industryRisk: industryRisk.score
    });

    return {
      entityId,
      entityName,
      entityType,
      screeningResults,
      ownershipStructure,
      transactionPatterns,
      geographicRisk,
      industryRisk,
      overallRisk,
      recommendation: this.generateRecommendation(overallRisk),
      timestamp: Date.now()
    };
  }

  private async investigateOwnership(entityId: string): Promise<any> {
    return {
      layers: 3,
      complexity: 0.65,
      ultimateBeneficialOwners: [],
      corporateStructure: [],
      jurisdictions: ['US', 'UK', 'Cayman Islands']
    };
  }

  private async analyzeTransactionPatterns(entityId: string): Promise<any> {
    return {
      totalTransactions: 1250,
      averageAmount: 15000,
      unusualPatterns: ['rapid_succession', 'round_amounts'],
      riskScore: 0.55
    };
  }

  private async assessGeographicRisk(entityId: string): Promise<any> {
    const highRiskCountries = ['AF', 'KP', 'IR', 'SY'];
    const mediumRiskCountries = ['PK', 'YE', 'LY'];

    return {
      countries: ['US', 'UK', 'PK'],
      highRiskExposure: false,
      mediumRiskExposure: true,
      score: 0.45
    };
  }

  private async assessIndustryRisk(entityId: string): Promise<any> {
    const highRiskIndustries = [
      'cryptocurrency', 'money_services', 'casinos',
      'precious_metals', 'real_estate'
    ];

    return {
      industry: 'financial_services',
      isHighRisk: false,
      score: 0.35
    };
  }

  private calculateOverallRisk(factors: any): number {
    const weights = {
      screeningMatches: 0.30,
      ownershipComplexity: 0.20,
      transactionRisk: 0.25,
      geographicRisk: 0.15,
      industryRisk: 0.10
    };

    let riskScore = 0;
    riskScore += (factors.screeningMatches > 0 ? 1.0 : 0.0) * weights.screeningMatches;
    riskScore += factors.ownershipComplexity * weights.ownershipComplexity;
    riskScore += factors.transactionRisk * weights.transactionRisk;
    riskScore += factors.geographicRisk * weights.geographicRisk;
    riskScore += factors.industryRisk * weights.industryRisk;

    return riskScore;
  }

  private generateRecommendation(riskScore: number): string {
    if (riskScore >= 0.75) {
      return 'REJECT - High risk entity. Do not onboard or continue business relationship.';
    } else if (riskScore >= 0.50) {
      return 'ENHANCED_DUE_DILIGENCE - Requires senior management approval and ongoing monitoring.';
    } else if (riskScore >= 0.30) {
      return 'STANDARD_DUE_DILIGENCE - Proceed with standard KYC procedures and periodic reviews.';
    } else {
      return 'SIMPLIFIED_DUE_DILIGENCE - Low risk entity. Standard onboarding procedures apply.';
    }
  }

  async updateSanctionsList(listId: string, entries: any[]): Promise<void> {
    const list = this.sanctionsLists.get(listId);
    if (list) {
      list.entries = entries;
      list.lastUpdated = Date.now();
      this.logger.info(`Updated sanctions list: ${listId} with ${entries.length} entries`);
    }
  }

  async updatePEPList(listId: string, entries: any[]): Promise<void> {
    const list = this.pepLists.get(listId);
    if (list) {
      list.entries = entries;
      list.lastUpdated = Date.now();
      this.logger.info(`Updated PEP list: ${listId} with ${entries.length} entries`);
    }
  }

  async generateSAR(
    entityId: string,
    suspiciousActivity: string,
    amount: number,
    details: any
  ): Promise<any> {
    const sarReport = {
      reportId: `SAR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      entityId,
      suspiciousActivity,
      amount,
      details,
      filingDate: new Date().toISOString(),
      status: 'PENDING_REVIEW',
      priority: amount > 100000 ? 'HIGH' : 'MEDIUM'
    };

    this.logger.info(`Generated SAR report: ${sarReport.reportId}`);
    return sarReport;
  }

  async performOngoingMonitoring(entityId: string): Promise<any> {
    const screeningResults = await this.screenEntity(entityId);
    const transactionMonitoring = await this.monitorTransactions(entityId);
    const behaviorAnalysis = await this.analyzeBehaviorChanges(entityId);

    const alerts = [];

    if (screeningResults.length > 0) {
      alerts.push({
        type: 'WATCHLIST_MATCH',
        severity: 'HIGH',
        details: screeningResults
      });
    }

    if (transactionMonitoring.anomalies.length > 0) {
      alerts.push({
        type: 'TRANSACTION_ANOMALY',
        severity: 'MEDIUM',
        details: transactionMonitoring.anomalies
      });
    }

    if (behaviorAnalysis.significantChanges) {
      alerts.push({
        type: 'BEHAVIOR_CHANGE',
        severity: 'MEDIUM',
        details: behaviorAnalysis
      });
    }

    return {
      entityId,
      monitoringDate: Date.now(),
      alerts,
      requiresReview: alerts.length > 0,
      nextReviewDate: Date.now() + (30 * 24 * 60 * 60 * 1000)
    };
  }

  private async monitorTransactions(entityId: string): Promise<any> {
    return {
      totalTransactions: 45,
      anomalies: [],
      riskScore: 0.25
    };
  }

  private async analyzeBehaviorChanges(entityId: string): Promise<any> {
    return {
      significantChanges: false,
      changes: [],
      riskScore: 0.15
    };
  }
}
