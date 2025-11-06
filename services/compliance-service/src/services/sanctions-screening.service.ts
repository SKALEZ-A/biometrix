import axios from 'axios';
import { logger } from '../../../packages/shared/src/utils/logger';

export interface SanctionsCheckResult {
  isMatch: boolean;
  matchScore: number;
  matchedEntities: SanctionedEntity[];
  checkDate: Date;
}

export interface SanctionedEntity {
  name: string;
  type: string;
  country: string;
  sanctionList: string;
  addedDate: Date;
}

export class SanctionsScreeningService {
  private sanctionsLists: Map<string, SanctionedEntity[]>;
  private apiEndpoint: string;

  constructor() {
    this.sanctionsLists = new Map();
    this.apiEndpoint = process.env.SANCTIONS_API_ENDPOINT || '';
    this.loadSanctionsLists();
  }

  private async loadSanctionsLists(): Promise<void> {
    try {
      const lists = ['OFAC', 'UN', 'EU', 'UK'];
      
      for (const list of lists) {
        const entities = await this.fetchSanctionsList(list);
        this.sanctionsLists.set(list, entities);
      }
      
      logger.info('Sanctions lists loaded successfully');
    } catch (error) {
      logger.error('Failed to load sanctions lists', { error });
    }
  }

  private async fetchSanctionsList(listName: string): Promise<SanctionedEntity[]> {
    // Fetch from external API or database
    return [];
  }

  async screenEntity(name: string, country?: string): Promise<SanctionsCheckResult> {
    const matches: SanctionedEntity[] = [];
    let highestScore = 0;

    for (const [listName, entities] of this.sanctionsLists) {
      for (const entity of entities) {
        const score = this.calculateMatchScore(name, entity.name);
        
        if (score > 0.8) {
          matches.push(entity);
          highestScore = Math.max(highestScore, score);
        }
      }
    }

    return {
      isMatch: matches.length > 0,
      matchScore: highestScore,
      matchedEntities: matches,
      checkDate: new Date()
    };
  }

  private calculateMatchScore(name1: string, name2: string): number {
    const normalized1 = name1.toLowerCase().trim();
    const normalized2 = name2.toLowerCase().trim();

    if (normalized1 === normalized2) return 1.0;

    const distance = this.levenshteinDistance(normalized1, normalized2);
    const maxLength = Math.max(normalized1.length, normalized2.length);
    
    return 1 - (distance / maxLength);
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

  async screenTransaction(transactionData: any): Promise<SanctionsCheckResult> {
    const senderCheck = await this.screenEntity(transactionData.senderName, transactionData.senderCountry);
    const receiverCheck = await this.screenEntity(transactionData.receiverName, transactionData.receiverCountry);

    return {
      isMatch: senderCheck.isMatch || receiverCheck.isMatch,
      matchScore: Math.max(senderCheck.matchScore, receiverCheck.matchScore),
      matchedEntities: [...senderCheck.matchedEntities, ...receiverCheck.matchedEntities],
      checkDate: new Date()
    };
  }
}
