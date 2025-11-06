import { ObjectId } from 'mongodb';

export interface IRole {
  _id?: ObjectId;
  roleId: string;
  name: string;
  description: string;
  permissions: string[];
  isSystem: boolean;
  isActive: boolean;
  hierarchy: number;
  metadata: RoleMetadata;
  createdAt: Date;
  updatedAt: Date;
}

export interface RoleMetadata {
  createdBy: string;
  updatedBy?: string;
  usersCount: number;
  category: 'admin' | 'user' | 'merchant' | 'analyst' | 'support';
}

export class Role implements IRole {
  _id?: ObjectId;
  roleId: string;
  name: string;
  description: string;
  permissions: string[];
  isSystem: boolean;
  isActive: boolean;
  hierarchy: number;
  metadata: RoleMetadata;
  createdAt: Date;
  updatedAt: Date;

  constructor(data: Partial<IRole>) {
    this._id = data._id;
    this.roleId = data.roleId || this.generateRoleId();
    this.name = data.name || '';
    this.description = data.description || '';
    this.permissions = data.permissions || [];
    this.isSystem = data.isSystem || false;
    this.isActive = data.isActive !== undefined ? data.isActive : true;
    this.hierarchy = data.hierarchy || 0;
    this.metadata = data.metadata || this.getDefaultMetadata();
    this.createdAt = data.createdAt || new Date();
    this.updatedAt = data.updatedAt || new Date();
  }

  private generateRoleId(): string {
    return `ROLE-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private getDefaultMetadata(): RoleMetadata {
    return {
      createdBy: 'system',
      usersCount: 0,
      category: 'user',
    };
  }

  hasPermission(permission: string): boolean {
    return this.permissions.includes(permission);
  }

  addPermission(permission: string): void {
    if (!this.permissions.includes(permission)) {
      this.permissions.push(permission);
      this.updatedAt = new Date();
    }
  }

  removePermission(permission: string): void {
    this.permissions = this.permissions.filter(p => p !== permission);
    this.updatedAt = new Date();
  }

  canBeDeleted(): boolean {
    return !this.isSystem && this.metadata.usersCount === 0;
  }
}
