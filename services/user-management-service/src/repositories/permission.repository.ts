import { MongoClient, Db, Collection } from 'mongodb';
import { logger } from '@shared/utils/logger';

export interface IPermission {
  permissionId: string;
  name: string;
  description: string;
  resource: string;
  action: string;
  category: string;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export class PermissionRepository {
  private db: Db | null = null;
  private collection: Collection<IPermission> | null = null;

  async connect(): Promise<void> {
    try {
      const client = await MongoClient.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017');
      this.db = client.db('user_management');
      this.collection = this.db.collection<IPermission>('permissions');
      
      await this.collection.createIndex({ permissionId: 1 }, { unique: true });
      await this.collection.createIndex({ resource: 1, action: 1 });
      await this.collection.createIndex({ category: 1 });
      
      logger.info('Permission repository connected');
    } catch (error) {
      logger.error('Error connecting permission repository:', error);
      throw error;
    }
  }

  async findAll(): Promise<IPermission[]> {
    if (!this.collection) await this.connect();
    return await this.collection!.find({ isActive: true }).toArray();
  }

  async findById(permissionId: string): Promise<IPermission | null> {
    if (!this.collection) await this.connect();
    return await this.collection!.findOne({ permissionId });
  }

  async findByCategory(category: string): Promise<IPermission[]> {
    if (!this.collection) await this.connect();
    return await this.collection!.find({ category, isActive: true }).toArray();
  }

  async findByResource(resource: string): Promise<IPermission[]> {
    if (!this.collection) await this.connect();
    return await this.collection!.find({ resource, isActive: true }).toArray();
  }

  async create(permission: IPermission): Promise<IPermission> {
    if (!this.collection) await this.connect();
    await this.collection!.insertOne(permission);
    return permission;
  }

  async update(permissionId: string, updates: Partial<IPermission>): Promise<IPermission | null> {
    if (!this.collection) await this.connect();
    const result = await this.collection!.findOneAndUpdate(
      { permissionId },
      { $set: updates },
      { returnDocument: 'after' }
    );
    return result.value;
  }

  async delete(permissionId: string): Promise<boolean> {
    if (!this.collection) await this.connect();
    const result = await this.collection!.deleteOne({ permissionId });
    return result.deletedCount > 0;
  }
}
