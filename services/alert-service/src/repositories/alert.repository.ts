import { MongoClient, Collection, ObjectId } from 'mongodb';
import { Alert, AlertStatus, AlertSeverity } from '../models/alert.model';

export class AlertRepository {
  private collection: Collection<Alert>;

  constructor(private mongoClient: MongoClient) {
    this.collection = this.mongoClient.db('fraud_prevention').collection<Alert>('alerts');
  }

  async create(alert: Omit<Alert, '_id' | 'createdAt' | 'updatedAt'>): Promise<Alert> {
    const now = new Date();
    const newAlert: Alert = {
      ...alert,
      _id: new ObjectId(),
      createdAt: now,
      updatedAt: now
    };

    await this.collection.insertOne(newAlert);
    return newAlert;
  }

  async findById(alertId: string): Promise<Alert | null> {
    return this.collection.findOne({ _id: new ObjectId(alertId) });
  }

  async findMany(filters: {
    status?: AlertStatus;
    severity?: AlertSeverity;
    type?: string;
    startDate?: Date;
    endDate?: Date;
    page: number;
    limit: number;
  }): Promise<{ alerts: Alert[]; total: number }> {
    const query: any = {};

    if (filters.status) query.status = filters.status;
    if (filters.severity) query.severity = filters.severity;
    if (filters.type) query.type = filters.type;
    if (filters.startDate || filters.endDate) {
      query.createdAt = {};
      if (filters.startDate) query.createdAt.$gte = filters.startDate;
      if (filters.endDate) query.createdAt.$lte = filters.endDate;
    }

    const skip = (filters.page - 1) * filters.limit;
    const alerts = await this.collection
      .find(query)
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(filters.limit)
      .toArray();

    const total = await this.collection.countDocuments(query);

    return { alerts, total };
  }

  async update(alertId: string, updates: Partial<Alert>): Promise<Alert | null> {
    const result = await this.collection.findOneAndUpdate(
      { _id: new ObjectId(alertId) },
      { $set: { ...updates, updatedAt: new Date() } },
      { returnDocument: 'after' }
    );

    return result.value;
  }

  async delete(alertId: string): Promise<boolean> {
    const result = await this.collection.deleteOne({ _id: new ObjectId(alertId) });
    return result.deletedCount > 0;
  }

  async getStatistics(filters?: {
    startDate?: Date;
    endDate?: Date;
  }): Promise<{
    total: number;
    byStatus: Record<AlertStatus, number>;
    bySeverity: Record<AlertSeverity, number>;
    byType: Record<string, number>;
  }> {
    const matchStage: any = {};
    if (filters?.startDate || filters?.endDate) {
      matchStage.createdAt = {};
      if (filters.startDate) matchStage.createdAt.$gte = filters.startDate;
      if (filters.endDate) matchStage.createdAt.$lte = filters.endDate;
    }

    const pipeline = [
      ...(Object.keys(matchStage).length > 0 ? [{ $match: matchStage }] : []),
      {
        $facet: {
          total: [{ $count: 'count' }],
          byStatus: [{ $group: { _id: '$status', count: { $sum: 1 } } }],
          bySeverity: [{ $group: { _id: '$severity', count: { $sum: 1 } } }],
          byType: [{ $group: { _id: '$type', count: { $sum: 1 } } }]
        }
      }
    ];

    const [result] = await this.collection.aggregate(pipeline).toArray();

    return {
      total: result.total[0]?.count || 0,
      byStatus: this.aggregateToRecord(result.byStatus),
      bySeverity: this.aggregateToRecord(result.bySeverity),
      byType: this.aggregateToRecord(result.byType)
    };
  }

  private aggregateToRecord(aggregateResult: any[]): Record<string, number> {
    return aggregateResult.reduce((acc, item) => {
      acc[item._id] = item.count;
      return acc;
    }, {});
  }
}
