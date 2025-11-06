import { MongoDBClient } from '../../../packages/shared/src/database/mongodb-client';
import { Role } from '../models/role.model';

export class RoleRepository {
  private db: MongoDBClient;
  private collection = 'roles';

  constructor() {
    this.db = new MongoDBClient();
  }

  public async create(role: Omit<Role, 'id' | 'createdAt' | 'updatedAt'>): Promise<Role> {
    const newRole: Role = {
      ...role,
      id: this.generateId(),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    await this.db.insertOne(this.collection, newRole);
    return newRole;
  }

  public async findById(id: string): Promise<Role | null> {
    return await this.db.findOne(this.collection, { id });
  }

  public async findByName(name: string): Promise<Role | null> {
    return await this.db.findOne(this.collection, { name });
  }

  public async findAll(): Promise<Role[]> {
    return await this.db.find(this.collection, {});
  }

  public async update(id: string, updates: Partial<Role>): Promise<Role | null> {
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

  public async addPermission(roleId: string, permissionId: string): Promise<Role | null> {
    await this.db.updateOne(
      this.collection,
      { id: roleId },
      { $addToSet: { permissions: permissionId }, $set: { updatedAt: new Date() } }
    );
    return await this.findById(roleId);
  }

  public async removePermission(roleId: string, permissionId: string): Promise<Role | null> {
    await this.db.updateOne(
      this.collection,
      { id: roleId },
      { $pull: { permissions: permissionId }, $set: { updatedAt: new Date() } }
    );
    return await this.findById(roleId);
  }

  private generateId(): string {
    return `role_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export const roleRepository = new RoleRepository();
