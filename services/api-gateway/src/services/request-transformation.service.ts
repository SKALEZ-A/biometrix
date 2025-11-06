import { Request, Response } from 'express';
import * as crypto from 'crypto';

interface TransformationRule {
  field: string;
  operation: 'rename' | 'remove' | 'transform' | 'add' | 'encrypt' | 'decrypt' | 'hash';
  target?: string;
  transformer?: (value: any) => any;
  condition?: (req: Request) => boolean;
}

interface TransformationConfig {
  requestRules?: TransformationRule[];
  responseRules?: TransformationRule[];
  preserveOriginal?: boolean;
}

export class RequestTransformationService {
  private config: TransformationConfig;
  private encryptionKey: Buffer;

  constructor(config: TransformationConfig = {}) {
    this.config = {
      preserveOriginal: false,
      ...config
    };
    this.encryptionKey = crypto.randomBytes(32);
  }

  transformRequest(req: Request): Request {
    if (!this.config.requestRules || this.config.requestRules.length === 0) {
      return req;
    }

    const transformed = this.config.preserveOriginal ? { ...req.body } : req.body;

    for (const rule of this.config.requestRules) {
      if (rule.condition && !rule.condition(req)) {
        continue;
      }

      this.applyRule(transformed, rule);
    }

    req.body = transformed;
    return req;
  }

  transformResponse(data: any, req: Request): any {
    if (!this.config.responseRules || this.config.responseRules.length === 0) {
      return data;
    }

    const transformed = this.config.preserveOriginal ? JSON.parse(JSON.stringify(data)) : data;

    for (const rule of this.config.responseRules) {
      if (rule.condition && !rule.condition(req)) {
        continue;
      }

      this.applyRule(transformed, rule);
    }

    return transformed;
  }

  private applyRule(obj: any, rule: TransformationRule): void {
    const value = this.getNestedValue(obj, rule.field);

    if (value === undefined && rule.operation !== 'add') {
      return;
    }

    switch (rule.operation) {
      case 'rename':
        if (rule.target) {
          this.setNestedValue(obj, rule.target, value);
          this.deleteNestedValue(obj, rule.field);
        }
        break;

      case 'remove':
        this.deleteNestedValue(obj, rule.field);
        break;

      case 'transform':
        if (rule.transformer) {
          const transformed = rule.transformer(value);
          this.setNestedValue(obj, rule.field, transformed);
        }
        break;

      case 'add':
        if (rule.transformer) {
          const newValue = rule.transformer(obj);
          this.setNestedValue(obj, rule.field, newValue);
        }
        break;

      case 'encrypt':
        const encrypted = this.encrypt(JSON.stringify(value));
        this.setNestedValue(obj, rule.field, encrypted);
        break;

      case 'decrypt':
        try {
          const decrypted = this.decrypt(value);
          this.setNestedValue(obj, rule.field, JSON.parse(decrypted));
        } catch (error) {
          console.error('Decryption failed:', error);
        }
        break;

      case 'hash':
        const hashed = crypto.createHash('sha256').update(String(value)).digest('hex');
        this.setNestedValue(obj, rule.field, hashed);
        break;
    }
  }

  private getNestedValue(obj: any, path: string): any {
    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[key];
    }

    return current;
  }

  private setNestedValue(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }

    current[keys[keys.length - 1]] = value;
  }

  private deleteNestedValue(obj: any, path: string): void {
    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current)) {
        return;
      }
      current = current[key];
    }

    delete current[keys[keys.length - 1]];
  }

  private encrypt(text: string): string {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', this.encryptionKey, iv);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return iv.toString('hex') + ':' + encrypted;
  }

  private decrypt(text: string): string {
    const parts = text.split(':');
    const iv = Buffer.from(parts[0], 'hex');
    const encryptedText = parts[1];
    const decipher = crypto.createDecipheriv('aes-256-cbc', this.encryptionKey, iv);
    let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  addRequestRule(rule: TransformationRule): void {
    if (!this.config.requestRules) {
      this.config.requestRules = [];
    }
    this.config.requestRules.push(rule);
  }

  addResponseRule(rule: TransformationRule): void {
    if (!this.config.responseRules) {
      this.config.responseRules = [];
    }
    this.config.responseRules.push(rule);
  }

  removeRequestRule(field: string): void {
    if (this.config.requestRules) {
      this.config.requestRules = this.config.requestRules.filter(rule => rule.field !== field);
    }
  }

  removeResponseRule(field: string): void {
    if (this.config.responseRules) {
      this.config.responseRules = this.config.responseRules.filter(rule => rule.field !== field);
    }
  }

  clearRules(): void {
    this.config.requestRules = [];
    this.config.responseRules = [];
  }
}

export class DataMaskingService extends RequestTransformationService {
  constructor() {
    super({
      responseRules: [
        {
          field: 'email',
          operation: 'transform',
          transformer: (email: string) => {
            if (!email || typeof email !== 'string') return email;
            const [local, domain] = email.split('@');
            if (!domain) return email;
            const maskedLocal = local.charAt(0) + '*'.repeat(local.length - 2) + local.charAt(local.length - 1);
            return `${maskedLocal}@${domain}`;
          }
        },
        {
          field: 'phone',
          operation: 'transform',
          transformer: (phone: string) => {
            if (!phone || typeof phone !== 'string') return phone;
            return phone.replace(/\d(?=\d{4})/g, '*');
          }
        },
        {
          field: 'ssn',
          operation: 'transform',
          transformer: (ssn: string) => {
            if (!ssn || typeof ssn !== 'string') return ssn;
            return '***-**-' + ssn.slice(-4);
          }
        },
        {
          field: 'creditCard',
          operation: 'transform',
          transformer: (cc: string) => {
            if (!cc || typeof cc !== 'string') return cc;
            return '**** **** **** ' + cc.slice(-4);
          }
        }
      ]
    });
  }

  maskSensitiveData(data: any, fields: string[]): any {
    const masked = JSON.parse(JSON.stringify(data));

    for (const field of fields) {
      const value = this.getNestedValue(masked, field);
      if (value !== undefined) {
        this.setNestedValue(masked, field, this.maskValue(value));
      }
    }

    return masked;
  }

  private maskValue(value: any): string {
    if (typeof value !== 'string') {
      value = String(value);
    }

    if (value.length <= 4) {
      return '*'.repeat(value.length);
    }

    return value.charAt(0) + '*'.repeat(value.length - 2) + value.charAt(value.length - 1);
  }

  private getNestedValue(obj: any, path: string): any {
    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[key];
    }

    return current;
  }

  private setNestedValue(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }

    current[keys[keys.length - 1]] = value;
  }
}

export const createTransformationService = (config?: TransformationConfig): RequestTransformationService => {
  return new RequestTransformationService(config);
};

export const createDataMaskingService = (): DataMaskingService => {
  return new DataMaskingService();
};
