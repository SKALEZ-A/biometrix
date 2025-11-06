import * as crypto from 'crypto';

export interface JWTPayload {
  sub: string;
  iat: number;
  exp: number;
  [key: string]: any;
}

export class JWTService {
  private static readonly ALGORITHM = 'RS256';

  static generate(
    payload: Omit<JWTPayload, 'iat' | 'exp'>,
    privateKey: string,
    expiresIn: number = 3600
  ): string {
    const now = Math.floor(Date.now() / 1000);
    
    const fullPayload: JWTPayload = {
      ...payload,
      iat: now,
      exp: now + expiresIn
    };
    
    const header = {
      alg: this.ALGORITHM,
      typ: 'JWT'
    };
    
    const encodedHeader = this.base64UrlEncode(JSON.stringify(header));
    const encodedPayload = this.base64UrlEncode(JSON.stringify(fullPayload));
    
    const signature = this.sign(`${encodedHeader}.${encodedPayload}`, privateKey);
    
    return `${encodedHeader}.${encodedPayload}.${signature}`;
  }

  static verify(token: string, publicKey: string): JWTPayload | null {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      
      const [encodedHeader, encodedPayload, signature] = parts;
      
      const isValid = this.verifySignature(
        `${encodedHeader}.${encodedPayload}`,
        signature,
        publicKey
      );
      
      if (!isValid) return null;
      
      const payload: JWTPayload = JSON.parse(this.base64UrlDecode(encodedPayload));
      
      const now = Math.floor(Date.now() / 1000);
      if (payload.exp && payload.exp < now) return null;
      
      return payload;
    } catch (error) {
      return null;
    }
  }

  static decode(token: string): JWTPayload | null {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      
      const payload = JSON.parse(this.base64UrlDecode(parts[1]));
      return payload;
    } catch (error) {
      return null;
    }
  }

  private static sign(data: string, privateKey: string): string {
    const sign = crypto.createSign('RSA-SHA256');
    sign.update(data);
    sign.end();
    return this.base64UrlEncode(sign.sign(privateKey));
  }

  private static verifySignature(
    data: string,
    signature: string,
    publicKey: string
  ): boolean {
    try {
      const verify = crypto.createVerify('RSA-SHA256');
      verify.update(data);
      verify.end();
      return verify.verify(publicKey, Buffer.from(this.base64UrlDecode(signature), 'base64'));
    } catch (error) {
      return false;
    }
  }

  private static base64UrlEncode(str: string): string {
    return Buffer.from(str)
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');
  }

  private static base64UrlDecode(str: string): string {
    str = str.replace(/-/g, '+').replace(/_/g, '/');
    while (str.length % 4) {
      str += '=';
    }
    return Buffer.from(str, 'base64').toString('utf8');
  }

  static refresh(token: string, privateKey: string, publicKey: string): string | null {
    const payload = this.verify(token, publicKey);
    if (!payload) return null;
    
    const { iat, exp, ...rest } = payload;
    return this.generate(rest, privateKey);
  }

  static isExpired(token: string): boolean {
    const payload = this.decode(token);
    if (!payload || !payload.exp) return true;
    
    const now = Math.floor(Date.now() / 1000);
    return payload.exp < now;
  }

  static getTimeToExpiry(token: string): number {
    const payload = this.decode(token);
    if (!payload || !payload.exp) return 0;
    
    const now = Math.floor(Date.now() / 1000);
    return Math.max(0, payload.exp - now);
  }
}
