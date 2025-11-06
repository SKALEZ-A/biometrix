import { MongoDBClient } from '../../../packages/shared/src/database/mongodb-client';
import { User } from '../models/user.model';

export class UserRepository {
  private db: MongoDBClient;
  private collection = 'users';

  constructor() {
    this.db = new MongoDBClient();
  }

  public async create(user: Omit<User, 'id' | 'createdAt' | 'updatedAt'>): Promise<User> {
    const newUser: User = {
      ...user,
      id: this.generateId(),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    await this.db.insertOne(this.collection, newUser);
    return newUser;
  }

  public async findById(id: string): Promise<User | null> {
    return await this.db.findOne(this.collection, { id });
  }

  public async findByEmail(email: string): Promise<User | null> {
    return await this.db.findOne(this.collection, { email });
  }

  public async findAll(filters: any = {}): Promise<User[]> {
    return await this.db.find(this.collection, filters);
  }

  public async update(id: string, updates: Partial<User>): Promise<User | null> {
    const updated = {
      ...updates,
      updatedAt: new Date()
    };

    await this.db.updateOne(this.collection, { id }, { $set: updated });
    return await this.findById(id);
  }

  public async delete(id: string): Promise<boolean> {
    const result = await this.db.deleteOne(this.collection, { id });
    return result.deletedCount > 0;
  }

  public async updateLastLogin(id: string): Promise<void> {
    await this.db.updateOne(
      this.collection,
      { id },
      { $set: { lastLoginAt: new Date() } }
    );
  }

  public async assignRole(userId: string, roleId: string): Promise<User | null> {
    await this.db.updateOne(
      this.collection,
      { id: userId },
      { $addToSet: { roles: roleId }, $set: { updatedAt: new Date() } }
    );
    return await this.findById(userId);
  }

  public async removeRole(userId: string, roleId: string): Promise<User | null> {
    await this.db.updateOne(
      this.collection,
      { id: userId },
      { $pull: { roles: roleId }, $set: { updatedAt: new Date() } }
    );
    return await this.findById(userId);
  }

  private generateId(): string {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export const userRepository = new UserRepository();
