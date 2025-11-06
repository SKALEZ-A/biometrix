export interface Permission {
  id: string;
  name: string;
  resource: string;
  action: 'create' | 'read' | 'update' | 'delete' | 'execute';
  description?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface PermissionGroup {
  id: string;
  name: string;
  permissions: string[];
  description?: string;
}

export class PermissionModel {
  public static readonly RESOURCES = {
    TRANSACTION: 'transaction',
    USER: 'user',
    BIOMETRIC: 'biometric',
    ALERT: 'alert',
    REPORT: 'report',
    MERCHANT: 'merchant',
    COMPLIANCE: 'compliance'
  };

  public static readonly ACTIONS = {
    CREATE: 'create',
    READ: 'read',
    UPDATE: 'update',
    DELETE: 'delete',
    EXECUTE: 'execute'
  };

  public static createPermissionKey(resource: string, action: string): string {
    return `${resource}:${action}`;
  }

  public static parsePermissionKey(key: string): { resource: string; action: string } {
    const [resource, action] = key.split(':');
    return { resource, action };
  }
}
