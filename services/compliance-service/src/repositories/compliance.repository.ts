import { MongoClient, Collection, ObjectId } from 'mongodb';
import { KYCVerification, AMLScreening, SanctionsCheck } from '../models/compliance.model';

export class ComplianceRepository {
  private kycCollection: Collection<KYCVerification>;
  private amlCollection: Collection<AMLScreening>;
  private sanctionsCollection: Collection<SanctionsCheck>;

  constructor(private mongoClient: MongoClient) {
    const db = this.mongoClient.db('fraud_prevention');
    this.kycCollection = db.collection<KYCVerification>('kyc_verifications');
    this.amlCollection = db.collection<AMLScreening>('aml_screenings');
    this.sanctionsCollection = db.collection<SanctionsCheck>('sanctions_checks');
  }

  async createKYCVerification(kyc: Omit<KYCVerification, '_id' | 'createdAt' | 'updatedAt'>): Promise<KYCVerification> {
    const now = new Date();
    const newKYC: KYCVerification = {
      ...kyc,
      _id: new ObjectId(),
      createdAt: now,
      updatedAt: now
    };

    await this.kycCollection.insertOne(newKYC);
    return newKYC;
  }

  async findKYCByUserId(userId: string): Promise<KYCVerification | null> {
    return this.kycCollection.findOne({ userId });
  }

  async updateKYCStatus(
    userId: string,
    status: KYCVerification['status'],
    updates: Partial<KYCVerification>
  ): Promise<KYCVerification | null> {
    const result = await this.kycCollection.findOneAndUpdate(
      { userId },
      { $set: { ...updates, status, updatedAt: new Date() } },
      { returnDocument: 'after' }
    );

    return result.value;
  }

  async createAMLScreening(screening: Omit<AMLScreening, '_id'>): Promise<AMLScreening> {
    const newScreening: AMLScreening = {
      ...screening,
      _id: new ObjectId()
    };

    await this.amlCollection.insertOne(newScreening);
    return newScreening;
  }

  async findAMLScreeningsByUserId(userId: string): Promise<AMLScreening[]> {
    return this.amlCollection
      .find({ userId })
      .sort({ screeningDate: -1 })
      .toArray();
  }

  async createSanctionsCheck(check: Omit<SanctionsCheck, '_id'>): Promise<SanctionsCheck> {
    const newCheck: SanctionsCheck = {
      ...check,
      _id: new ObjectId()
    };

    await this.sanctionsCollection.insertOne(newCheck);
    return newCheck;
  }

  async findSanctionsChecksByEntityId(entityId: string): Promise<SanctionsCheck[]> {
    return this.sanctionsCollection
      .find({ entityId })
      .sort({ checkDate: -1 })
      .toArray();
  }

  async getComplianceStatistics(): Promise<{
    totalKYC: number;
    approvedKYC: number;
    pendingKYC: number;
    amlHighRisk: number;
    sanctionsMatches: number;
  }> {
    const [kycStats, amlStats, sanctionsStats] = await Promise.all([
      this.kycCollection.aggregate([
        {
          $group: {
            _id: '$status',
            count: { $sum: 1 }
          }
        }
      ]).toArray(),
      this.amlCollection.countDocuments({ riskLevel: 'high' }),
      this.sanctionsCollection.countDocuments({ status: 'match_found' })
    ]);

    const kycByStatus = kycStats.reduce((acc, item) => {
      acc[item._id] = item.count;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalKYC: Object.values(kycByStatus).reduce((sum, count) => sum + count, 0),
      approvedKYC: kycByStatus['approved'] || 0,
      pendingKYC: kycByStatus['pending_review'] || 0,
      amlHighRisk: amlStats,
      sanctionsMatches: sanctionsStats
    };
  }
}
