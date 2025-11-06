import { logger } from '@shared/utils/logger';

export interface Permission {
  id: string;
  name: string;
  description: string;
  resource: string;
  action: string;
  category: string;
}

export class PermissionService {
  private permissions: Map<string, Permission> = new Map();

  constructor() {
    this.initializePermissions();
  }

  private initializePermissions(): void {
    const defaultPermissions: Permission[] = [
      // User Management
      { id: 'user:read', name: 'Read Users', description: 'View user information', resource: 'user', action: 'read', category: 'user-management' },
      { id: 'user:create', name: 'Create Users', description: 'Create new users', resource: 'user', action: 'create', category: 'user-management' },
      { id: 'user:update', name: 'Update Users', description: 'Modify user information', resource: 'user', action: 'update', category: 'user-management' },
      { id: 'user:delete', name: 'Delete Users', description: 'Remove users', resource: 'user', action: 'delete', category: 'user-management' },
      
      // Role Management
      { id: 'role:read', name: 'Read Roles', description: 'View role information', resource: 'role', action: 'read', category: 'role-management' },
      { id: 'role:create', name: 'Create Roles', description: 'Create new roles', resource: 'role', action: 'create', category: 'role-management' },
      { id: 'role:update', name: 'Update Roles', description: 'Modify role information', resource: 'role', action: 'update', category: 'role-management' },
      { id: 'role:delete', name: 'Delete Roles', description: 'Remove roles', resource: 'role', action: 'delete', category: 'role-management' },
      
      // Transaction Management
      { id: 'transaction:read', name: 'Read Transactions', description: 'View transaction data', resource: 'transaction', action: 'read', category: 'transaction' },
      { id: 'transaction:create', name: 'Create Transactions', description: 'Process transactions', resource: 'transaction', action: 'create', category: 'transaction' },
      { id: 'transaction:update', name: 'Update Transactions', description: 'Modify transactions', resource: 'transaction', action: 'update', category: 'transaction' },
      { id: 'transaction:refund', name: 'Refund Transactions', description: 'Process refunds', resource: 'transaction', action: 'refund', category: 'transaction' },
      
      // Fraud Detection
      { id: 'fraud:read', name: 'Read Fraud Data', description: 'View fraud detection results', resource: 'fraud', action: 'read', category: 'fraud-detection' },
      { id: 'fraud:analyze', name: 'Analyze Fraud', description: 'Run fraud analysis', resource: 'fraud', action: 'analyze', category: 'fraud-detection' },
      { id: 'fraud:review', name: 'Review Fraud Cases', description: 'Review flagged cases', resource: 'fraud', action: 'review', category: 'fraud-detection' },
      { id: 'fraud:override', name: 'Override Fraud Decisions', description: 'Override automated decisions', resource: 'fraud', action: 'override', category: 'fraud-detection' },
      
      // Biometric Management
      { id: 'biometric:read', name: 'Read Biometric Data', description: 'View biometric information', resource: 'biometric', action: 'read', category: 'biometric' },
      { id: 'biometric:enroll', name: 'Enroll Biometrics', description: 'Register biometric data', resource: 'biometric', action: 'enroll', category: 'biometric' },
      { id: 'biometric:verify', name: 'Verify Biometrics', description: 'Verify biometric authentication', resource: 'biometric', action: 'verify', category: 'biometric' },
      { id: 'biometric:delete', name: 'Delete Biometric Data', description: 'Remove biometric records', resource: 'biometric', action: 'delete', category: 'biometric' },
      
      // Alert Management
      { id: 'alert:read', name: 'Read Alerts', description: 'View alerts', resource: 'alert', action: 'read', category: 'alert' },
      { id: 'alert:create', name: 'Create Alerts', description: 'Generate alerts', resource: 'alert', action: 'create', category: 'alert' },
      { id: 'alert:update', name: 'Update Alerts', description: 'Modify alert status', resource: 'alert', action: 'update', category: 'alert' },
      { id: 'alert:resolve', name: 'Resolve Alerts', description: 'Mark alerts as resolved', resource: 'alert', action: 'resolve', category: 'alert' },
      
      // Compliance
      { id: 'compliance:read', name: 'Read Compliance Data', description: 'View compliance information', resource: 'compliance', action: 'read', category: 'compliance' },
      { id: 'compliance:kyc', name: 'KYC Verification', description: 'Perform KYC checks', resource: 'compliance', action: 'kyc', category: 'compliance' },
      { id: 'compliance:aml', name: 'AML Screening', description: 'Perform AML screening', resource: 'compliance', action: 'aml', category: 'compliance' },
      { id: 'compliance:report', name: 'Generate Compliance Reports', description: 'Create compliance reports', resource: 'compliance', action: 'report', category: 'compliance' },
      
      // Analytics
      { id: 'analytics:read', name: 'Read Analytics', description: 'View analytics data', resource: 'analytics', action: 'read', category: 'analytics' },
      { id: 'analytics:export', name: 'Export Analytics', description: 'Export analytics reports', resource: 'analytics', action: 'export', category: 'analytics' },
      
      // System Administration
      { id: 'system:config', name: 'System Configuration', description: 'Modify system settings', resource: 'system', action: 'config', category: 'system' },
      { id: 'system:audit', name: 'View Audit Logs', description: 'Access audit trails', resource: 'system', action: 'audit', category: 'system' },
      { id: 'system:monitor', name: 'System Monitoring', description: 'Monitor system health', resource: 'system', action: 'monitor', category: 'system' },
    ];

    defaultPermissions.forEach(permission => {
      this.permissions.set(permission.id, permission);
    });

    logger.info(`Initialized ${this.permissions.size} permissions`);
  }

  getAllPermissions(): Permission[] {
    return Array.from(this.permissions.values());
  }

  getPermissionById(id: string): Permission | undefined {
    return this.permissions.get(id);
  }

  getPermissionsByCategory(category: string): Permission[] {
    return Array.from(this.permissions.values()).filter(p => p.category === category);
  }

  getPermissionsByResource(resource: string): Permission[] {
    return Array.from(this.permissions.values()).filter(p => p.resource === resource);
  }

  validatePermissions(permissionIds: string[]): { valid: string[]; invalid: string[] } {
    const valid: string[] = [];
    const invalid: string[] = [];

    permissionIds.forEach(id => {
      if (this.permissions.has(id)) {
        valid.push(id);
      } else {
        invalid.push(id);
      }
    });

    return { valid, invalid };
  }

  hasPermission(userPermissions: string[], requiredPermission: string): boolean {
    return userPermissions.includes(requiredPermission);
  }

  hasAnyPermission(userPermissions: string[], requiredPermissions: string[]): boolean {
    return requiredPermissions.some(permission => userPermissions.includes(permission));
  }

  hasAllPermissions(userPermissions: string[], requiredPermissions: string[]): boolean {
    return requiredPermissions.every(permission => userPermissions.includes(permission));
  }
}

export const permissionService = new PermissionService();
